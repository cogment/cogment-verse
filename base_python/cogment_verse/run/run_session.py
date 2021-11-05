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

log = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-statements

TRIALLAUNCHER_TRIAL_RUNNING_LEN = Gauge("triallauncher_trial_running_len", "Length of the running trials")
TRIALLAUNCHER_SAMPLE_CONSUMED_COUNTER = Counter("triallauncher_sample_consumed", "Counter of consumed samples")
TRIALLAUNCHER_SAMPLE_QUEUE_LEN = Gauge("triallauncher_sample_queue_len", "Length of the sample queue")
TRIALLAUNCHER_SAMPLE_PRODUCED_COUNTER = Counter("triallauncher_sample_produced", "Counter of produced samples")
TRIALLAUNCHER_TRIAL_TIME = Summary("triallauncher_trial_seconds", "Time spent running trials")
TRIALLAUNCHER_TRIAL_STARTED_COUNTER = Counter("triallauncher_trial_started", "Counter of started trials")
TRIALLAUNCHER_START_TRIAL_TIME = Summary("triallauncher_start_trial_seconds", "Time spent starting trials")


def default_on_progress(_launched_trial_count, _finished_trial_count):
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

    async def _start_trials(self, trial_configs, sample_queue):
        def produce_training_sample(trial_id, tick_id, sample):
            step_id, step_timestamp = self._stepper.step(trial_id, tick_id)
            sample_queue.put_nowait((step_id, step_timestamp, trial_id, tick_id, sample))

            TRIALLAUNCHER_SAMPLE_PRODUCED_COUNTER.inc()

            return step_id, step_timestamp

        async def trials_samples_listener(trial_ids):
            trial_infos = await self._trial_datastore_client.retrieve_trials(trial_ids)

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
            sample_generator = await self._trial_datastore_client.retrieve_samples(trial_ids)
            async for sample in sample_generator():
                await run_sample_producer_sessions[sample.trial_id].on_trial_sample(sample)

            for session in run_sample_producer_sessions.values():
                await session.on_trial_done()

            await asyncio.wait(run_sample_producer_tasks)

            trial_duration_seconds = time.time() - trial_start_time
            TRIALLAUNCHER_TRIAL_TIME.observe(trial_duration_seconds)

        done, _ = await asyncio.wait(
            [self._controller.start_trial(trial_config=trial_config) for trial_config in trial_configs]
        )

        trial_ids = [r.result() for r in done]

        asyncio.create_task(trials_samples_listener(trial_ids))

        return trial_ids

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
            # We don't want terminate to fail, exeception handling is dealt with in get_status().
            pass

        return self.get_status()

    def count_steps(self):
        return self._stepper.count_steps()

    async def start_trials_and_wait_for_termination(
        self, trial_configs, max_parallel_trials=4, on_progress=default_on_progress
    ):
        if self.get_status() is not RunSessionStatus.RUNNING:
            raise RuntimeError(f"[{self.run_id}] not running")

        sample_queue = asyncio.Queue()

        async def run_trials():
            try:
                running_trial_ids = set()
                trial_len = len(trial_configs)

                # Initialize the end-of-trial event generator
                watch_trials_end = self._controller.watch_trials(trial_state_filters=[cogment.TrialState.ENDED])

                # Start a first chunk of trials
                initial_trial_configs_chunk = trial_configs[0:max_parallel_trials]

                on_progress(len(initial_trial_configs_chunk), 0)
                with TRIALLAUNCHER_START_TRIAL_TIME.time():
                    trial_ids = await self._start_trials(
                        initial_trial_configs_chunk,
                        sample_queue,
                    )
                running_trial_ids.update(trial_ids)
                TRIALLAUNCHER_TRIAL_RUNNING_LEN.set(len(running_trial_ids))

                next_trial_idx = max_parallel_trials
                async for trial_info in watch_trials_end:
                    running_trial_ids.discard(trial_info.trial_id)
                    running_trials_count = len(running_trial_ids)

                    if self._terminating:
                        # The run is terminating let's stop starting trials
                        raise asyncio.CancelledError()

                    if running_trials_count >= max_parallel_trials:
                        # Nothing to do, a trial we didn't start just ended
                        continue

                    if next_trial_idx < trial_len:
                        # Available trial slots and some trials to launch
                        trial_configs_chunk = trial_configs[
                            next_trial_idx : next_trial_idx + max_parallel_trials - running_trials_count
                        ]

                        on_progress(
                            next_trial_idx,
                            next_trial_idx + max_parallel_trials - running_trials_count,
                        )
                        with TRIALLAUNCHER_START_TRIAL_TIME.time():
                            trial_ids = await self._start_trials(
                                trial_configs_chunk,
                                sample_queue,
                            )
                        running_trial_ids.update(trial_ids)
                        TRIALLAUNCHER_TRIAL_RUNNING_LEN.set(len(running_trial_ids))

                        next_trial_idx += len(trial_ids)
                    elif running_trials_count > 0:
                        # Waiting for the remaing trials to finish
                        continue
                    else:
                        # We are done
                        break
            except asyncio.CancelledError:
                raise
            except Exception as error:
                log.error(
                    f"[{self.run_id}] Unexpected error while running {len(trial_configs)} trials",
                    exc_info=error,
                )
                raise

        run_trials_task = asyncio.create_task(run_trials())

        while not run_trials_task.done() or not sample_queue.empty():
            full_sample = await sample_queue.get()
            yield full_sample

            TRIALLAUNCHER_SAMPLE_CONSUMED_COUNTER.inc()
            TRIALLAUNCHER_SAMPLE_QUEUE_LEN.set(sample_queue.qsize())

        # Awaiting `run_trials_task` to rethrow exceptions
        await run_trials_task
