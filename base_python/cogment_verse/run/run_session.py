# Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cogment_verse.run.run_stepper import RunStepper
from cogment_verse.run.run_sample_producer_session import RunSampleProducerSession

import cogment
from prometheus_client import Counter, Gauge, Summary
from names_generator import generate_name

import asyncio
from datetime import datetime
import time
from enum import Enum, auto
import logging
import queue

log = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-statements

TRIALLAUNCHER_TRIAL_RUNNING_LEN = Gauge("triallauncher_trial_running_len", "Length of the running trials")
TRIALLAUNCHER_SAMPLE_CONSUMED_COUNTER = Counter("triallauncher_sample_consumed", "Counter of consumed samples")
TRIALLAUNCHER_SAMPLE_QUEUE_LEN = Gauge("triallauncher_sample_queue_len", "Length of the sample queue")
TRIALLAUNCHER_SAMPLE_PRODUCED_COUNTER = Counter("triallauncher_sample_produced", "Counter of produced samples")
TRIALLAUNCHER_TRIAL_TIME = Summary("triallauncher_trial_seconds", "Time spent running trials")
TRIALLAUNCHER_TRIAL_STARTED_COUNTER = Counter("triallauncher_trial_started", "Counter of started trials")
TRIALLAUNCHER_START_TRIAL_TIME = Summary("triallauncher_start_trial_seconds", "Time spent starting trials")

PRIO_QUEUE_HIGH_PRIO = 0
PRIO_QUEUE_LOW_PRIO = 1
PRIO_QUEUE_END_PRIO = 2


def default_on_progress(_launched_trials_count, _finished_trials_count):
    pass


class RunSessionStatus(Enum):
    CREATED = auto()
    RUNNING = auto()
    TERMINATING = auto()
    TERMINATED = auto()
    SUCCESS = auto()
    ERROR = auto()


class RunSession:
    def __init__(
        self,
        cog_settings,
        controller,
        trial_datastore_client,
        config,
        run_sample_producer_impl,
        impl_name,
        run_impl,
        params_name,
        run_id=None,
    ):
        super().__init__()

        self.run_id = run_id if run_id is not None else generate_name()
        self.config = config
        self.impl_name = impl_name
        self.params_name = params_name
        self.start_time = datetime.now()

        self._cog_settings = cog_settings
        self._run_sample_producer_impl = run_sample_producer_impl
        self._controller = controller
        self._trial_datastore_client = trial_datastore_client
        self._stepper = RunStepper()

        self._run_impl = run_impl
        self._task = None
        self._terminating = False

    def get_status(self):
        if self._task is None:
            return RunSessionStatus.CREATED
        try:
            self._task.result()
            return RunSessionStatus.SUCCESS
        except asyncio.CancelledError:
            return RunSessionStatus.TERMINATED
        except asyncio.InvalidStateError:
            if self._terminating:
                return RunSessionStatus.TERMINATING
            return RunSessionStatus.RUNNING
        except Exception as error:
            log.error(f"[{self.run_id}] Uncaught error occured during the run", exc_info=error)
            return RunSessionStatus.ERROR

    def exec(self):
        if self.get_status() is not RunSessionStatus.CREATED:
            raise RuntimeError(f"[{self.run_id}] already started")

        async def exec_run():
            log.info(f"[{self.run_id}] Starting run...")
            impl_task = asyncio.create_task(self._run_impl(self))
            try:
                await impl_task
                log.info(f"[{self.run_id}] Run suceeded")
            except asyncio.CancelledError:
                log.info(f"[{self.run_id}] Terminating run...")
                self._terminating = True
                try:
                    await impl_task
                except asyncio.CancelledError:
                    pass
                log.info(f"[{self.run_id}] Run terminated")
                raise
            except Exception as error:
                log.error(
                    f"[{self.run_id}] Uncaught error occured during the run",
                    exc_info=error,
                )
                raise error

        self._task = asyncio.create_task(exec_run())
        return self._task

    async def terminate(self):
        if self.get_status() is not RunSessionStatus.RUNNING:
            raise RuntimeError(f"[{self.run_id}] not running")

        self._task.cancel()
        try:
            await self._task
        except Exception:
            # We don't want terminate to fail, exception handling is dealt with in get_status().
            pass

        return self.get_status()

    def count_steps(self):
        return self._stepper.count_steps()

    @staticmethod
    async def _do_enqueue_trial_configs(trial_config_queue_out, trial_configs):
        for trial_config in trial_configs:
            await trial_config_queue_out.put(trial_config)

        # Making sure the consumer knowns that it's finished
        await trial_config_queue_out.put(None)

    async def _do_start_trials(
        self,
        trial_configs_queue_in,
        trial_ids_chunk_prio_queue_out,
        max_parallel_trials,
        on_progress,
        start_trial_throttle_timeout,
    ):
        launched_trials_count = 0
        finished_trials_count = 0

        running_trial_ids = set()

        async def monitor_ended_trials():
            nonlocal finished_trials_count
            async for ended_trial_info in self._controller.watch_trials(trial_state_filters=[cogment.TrialState.ENDED]):
                if ended_trial_info.trial_id in running_trial_ids:
                    log.debug(f"[{self.run_id}] Trial [{ended_trial_info.trial_id}] ended")
                    running_trial_ids.discard(ended_trial_info.trial_id)
                    finished_trials_count += 1

        try:
            monitor_ended_trials_task = asyncio.create_task(monitor_ended_trials())

            while True:
                if monitor_ended_trials_task.cancelled():
                    raise asyncio.CancelledError()
                if monitor_ended_trials_task.done():
                    # The only way `monitor_ended_trials_task` finishes is if an error occured
                    raise RuntimeError(
                        "An error occured while monitoring the trials"
                    ) from monitor_ended_trials_task.exception()

                trial_slots_count = max_parallel_trials - len(running_trial_ids)
                on_progress(launched_trials_count, finished_trials_count)

                if trial_slots_count > 0:
                    to_start_trial_configs = []
                    done = False
                    while len(to_start_trial_configs) < trial_slots_count:
                        trial_config = await trial_configs_queue_in.get()
                        if trial_config is None:
                            done = True
                            break
                        to_start_trial_configs.append(trial_config)

                    if len(to_start_trial_configs) > 0:
                        log.debug(f"[{self.run_id}] Starting {len(to_start_trial_configs)} trials")

                        with TRIALLAUNCHER_START_TRIAL_TIME.time():
                            done_start_trial_tasks, _ = await asyncio.wait(
                                [
                                    self._controller.start_trial(trial_config=trial_config)
                                    for trial_config in to_start_trial_configs
                                ]
                            )
                            started_trial_ids = {r.result() for r in done_start_trial_tasks}

                        log.debug(f"[{self.run_id}] Trials [{', '.join(started_trial_ids)}] started")

                        launched_trials_count += len(started_trial_ids)
                        running_trial_ids.update(started_trial_ids)
                        await trial_ids_chunk_prio_queue_out.put((PRIO_QUEUE_LOW_PRIO, started_trial_ids))

                        for trial_config in to_start_trial_configs:
                            trial_configs_queue_in.task_done()

                    if done:
                        # Making sure the consumer knowns that it's finished
                        await trial_ids_chunk_prio_queue_out.put((PRIO_QUEUE_END_PRIO, None))
                        break

                else:
                    # at least `max_parallel_trials` currently running, let's wait a little bit before checking again
                    timeout_seconds = start_trial_throttle_timeout * (1 / 1000.0)
                    log.debug(
                        f"[{self.run_id}] Waiting {timeout_seconds:.1f} seconds before checking if trials need to be restarted"
                    )
                    await asyncio.sleep(timeout_seconds)

        except asyncio.CancelledError:
            # Cancelling subtasks
            monitor_ended_trials_task.cancel()
            await monitor_ended_trials_task

    async def _do_observe_trials(self, trial_ids_chunk_prio_queue_in, sample_queue_out, trial_datastore_timeout):
        def produce_training_sample(trial_id, tick_id, sample):
            step_id, step_timestamp = self._stepper.step(trial_id, tick_id)
            sample_queue_out.put_nowait((step_id, step_timestamp, trial_id, tick_id, sample))

            TRIALLAUNCHER_SAMPLE_PRODUCED_COUNTER.inc()

            return step_id, step_timestamp

        async def observe_trial_ids_chunk(trial_ids):
            # Retrieve trial infos from the trial datastore
            trial_infos = await self._trial_datastore_client.retrieve_trials(trial_ids, trial_datastore_timeout)
            # `trial_infos` only contains info related to the trials already known to the trial datastore
            known_trial_ids = {trial_info.trial_id for trial_info in trial_infos}
            unknown_trials_ids = set(trial_ids).difference(known_trial_ids)

            if len(unknown_trials_ids) > 0:
                log.info(
                    f"[{self.run_id}] Trials [{', '.join(unknown_trials_ids)}] didn't start generating data under {trial_datastore_timeout}ms, retrying"
                )
                trial_ids_chunk_prio_queue_in.put_nowait((PRIO_QUEUE_HIGH_PRIO, unknown_trials_ids))

            if len(known_trial_ids) == 0:
                return

            run_sample_producer_sessions = {
                trial_info.trial_id: RunSampleProducerSession(
                    cog_settings=self._cog_settings,
                    run_id=self.run_id,
                    trial_id=trial_info.trial_id,
                    trial_params=trial_info.params,
                    produce_training_sample=produce_training_sample,
                    run_config=self.config,
                    run_sample_producer_impl=self._run_sample_producer_impl,
                )
                for trial_info in trial_infos
            }

            run_sample_producer_tasks = [session.exec() for session in run_sample_producer_sessions.values()]
            TRIALLAUNCHER_TRIAL_STARTED_COUNTER.inc(len(run_sample_producer_tasks))

            trial_start_time = time.time()
            sample_generator = await self._trial_datastore_client.retrieve_samples(known_trial_ids)
            async for sample in sample_generator():
                await run_sample_producer_sessions[sample.trial_id].on_trial_sample(sample)

            for session in run_sample_producer_sessions.values():
                await session.on_trial_done()

            await asyncio.wait(run_sample_producer_tasks)

            trial_duration_seconds = time.time() - trial_start_time
            TRIALLAUNCHER_TRIAL_TIME.observe(trial_duration_seconds)

            trial_ids_chunk_prio_queue_in.task_done()

        observe_tasks = []
        try:
            while True:
                done_observe_tasks = [t for t in observe_tasks if t.done()]
                cancelled_observe_tasks = [t for t in done_observe_tasks if t.cancelled()]
                if len(cancelled_observe_tasks) > 0:
                    raise asyncio.CancelledError() from cancelled_observe_tasks[0].exception()
                failed_observe_tasks = [t for t in done_observe_tasks if t.exception() is not None]
                if len(failed_observe_tasks) > 0:
                    raise RuntimeError("An error occured while observing the trials log") from failed_observe_tasks[
                        0
                    ].exception()

                _, trial_ids = await trial_ids_chunk_prio_queue_in.get()
                if trial_ids is None:
                    break

                observe_tasks.append(asyncio.create_task(observe_trial_ids_chunk(trial_ids)))
        except asyncio.CancelledError:
            # Cancelling subtasks
            observe_tasks_gathering = asyncio.gather(*observe_tasks)
            observe_tasks_gathering.cancel()
            await observe_tasks_gathering

    async def start_trials_and_wait_for_termination(
        self, trial_configs, max_parallel_trials=4, on_progress=default_on_progress
    ):
        if self.get_status() is not RunSessionStatus.RUNNING:
            raise RuntimeError(f"[{self.run_id}] not running")

        trial_config_queue = asyncio.Queue()
        started_trial_ids_chunk_prio_queue = asyncio.PriorityQueue()
        sample_queue = asyncio.Queue()

        enqueue_trial_configs = asyncio.create_task(self._do_enqueue_trial_configs(trial_config_queue, trial_configs))
        start_trials = asyncio.create_task(
            self._do_start_trials(
                trial_config_queue,
                started_trial_ids_chunk_prio_queue,
                max_parallel_trials,
                on_progress,
                start_trial_throttle_timeout=500,
            )
        )
        observe_trials = asyncio.create_task(
            self._do_observe_trials(started_trial_ids_chunk_prio_queue, sample_queue, trial_datastore_timeout=5000)
        )
        workers = asyncio.gather(enqueue_trial_configs, start_trials, observe_trials)

        try:
            # We don't want the workers to be cancelled everytime a sample is retrieved
            shielded_workers = asyncio.shield(workers)

            while not (sample_queue.empty() and workers.done()):
                get_next_sample = asyncio.create_task(sample_queue.get())
                done, _ = await asyncio.wait({get_next_sample, shielded_workers}, return_when=asyncio.FIRST_COMPLETED)

                if get_next_sample in done:
                    yield get_next_sample.result()
                    sample_queue.task_done()

                    TRIALLAUNCHER_SAMPLE_CONSUMED_COUNTER.inc()
                    TRIALLAUNCHER_SAMPLE_QUEUE_LEN.set(sample_queue.qsize())

                if shielded_workers in done:
                    err = workers.exception()
                    if err is None:
                        # Workers have finished, trials were ran and listened to
                        # Let's continue as long as there's still samples to emit
                        continue

                    if err is asyncio.CancelledError:
                        raise asyncio.CancelledError

                    raise RuntimeError(
                        f"[{self.run_id}] error while running and listening for trials"
                    ) from workers.exception()
        finally:
            # Watever happens we want to cancel those workers when the function's returns
            workers.cancel()
            await workers
