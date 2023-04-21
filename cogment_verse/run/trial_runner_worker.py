# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import asyncio
import logging
import sys
from multiprocessing import Process

import cogment

from ..services_directory import ServiceType

log = logging.getLogger(__name__)


class TrialStartedQueueEvent:
    def __init__(self, trial_id=None, trial_idx=None, done=False):
        self.trial_idx = trial_idx
        self.trial_id = trial_id
        self.done = done


class TrialEndedQueueEvent:
    def __init__(self, trial_id=None, trial_idx=None, done=False):
        self.trial_idx = trial_idx
        self.trial_id = trial_id
        self.done = done


async def async_trial_runner_worker(
    trials_id_and_params, services_directory, trial_started_queue, trial_ended_queue, num_parallel_trials
):
    # Importing 'specs' only in the subprocess (i.e. where generate has been properly executed)
    # pylint: disable-next=import-outside-toplevel
    from cogment_verse.specs import cog_settings

    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_verse_trial_runner")

    controller = await services_directory.get_controller(context)

    num_trials = len(trials_id_and_params)
    num_started_trials = 0
    num_ended_trials = 0
    running_trials = {}

    async def start_trials():
        nonlocal running_trials
        nonlocal num_started_trials
        num_to_start_trials = num_parallel_trials - len(running_trials)
        if num_to_start_trials <= 0:
            return

        trials_id_and_params_chunk = trials_id_and_params[
            num_started_trials : min(num_trials, num_started_trials + num_to_start_trials)
        ]
        if len(trials_id_and_params_chunk) == 0:
            return

        for (trial_id, serialized_trial_params) in trials_id_and_params_chunk:
            trial_params = cogment.TrialParameters(cog_settings)
            trial_params.deserialize(serialized_trial_params)

            # Retrieve the endpoints from the services directory
            trial_params.environment_endpoint = services_directory.get(
                ServiceType.ENVIRONMENT, trial_params.environment_implementation
            )
            trial_params.datalog_endpoint = services_directory.get(ServiceType.TRIAL_DATASTORE)
            for actor in trial_params.actors:
                actor.endpoint = services_directory.get(ServiceType.ACTOR, actor.implementation)

            actual_trial_id = await controller.start_trial(trial_id_requested=trial_id, trial_params=trial_params)
            if actual_trial_id is None:
                raise RuntimeError(f"Unable to start a trial with id [{trial_id}]")
            trial_idx = num_started_trials
            running_trials[trial_id] = trial_idx
            num_started_trials += 1
            log.debug(f"Trial [{trial_id}] started, {num_trials-num_started_trials} trials remaining to start.")
            if trial_started_queue is not None:
                trial_started_queue.put(TrialStartedQueueEvent(trial_id=trial_id, trial_idx=trial_idx))

    async def await_trials():
        nonlocal running_trials
        nonlocal num_ended_trials
        async for trial_info in controller.watch_trials(trial_state_filters=[cogment.TrialState.ENDED]):
            if trial_info.trial_id in running_trials:
                trial_idx = running_trials[trial_info.trial_id]
                del running_trials[trial_info.trial_id]
                num_ended_trials += 1
                if trial_ended_queue is not None:
                    trial_ended_queue.put(TrialEndedQueueEvent(trial_id=trial_info.trial_id, trial_idx=trial_idx))
                log.debug(f"Trial [{trial_info.trial_id}] ended, {num_trials-num_ended_trials} trials remaining.")
                if num_ended_trials == num_trials:
                    break
                if len(running_trials) < num_parallel_trials:
                    await start_trials()

    try:
        await_trials_start = asyncio.create_task(await_trials())
        await start_trials()
        await await_trials_start
    finally:
        if trial_started_queue is not None:
            trial_started_queue.put(TrialEndedQueueEvent(done=True))
        if trial_ended_queue is not None:
            trial_ended_queue.put(TrialEndedQueueEvent(done=True))


def trial_runner_worker(
    trials_id_and_params, services_directory, trial_started_queue, trial_ended_queue, num_parallel_trials
):
    try:
        asyncio.run(
            async_trial_runner_worker(
                trials_id_and_params, services_directory, trial_started_queue, trial_ended_queue, num_parallel_trials
            )
        )
    except KeyboardInterrupt:
        sys.exit(-1)


def start_trial_runner_worker(
    trials_id_and_params, services_directory, trial_started_queue=None, trial_ended_queue=None, num_parallel_trials=10
):
    worker = Process(
        name="trial_runner_worker",
        target=trial_runner_worker,
        args=(
            [(trial_id, trial_params.serialize()) for (trial_id, trial_params) in trials_id_and_params],
            services_directory,
            trial_started_queue,
            trial_ended_queue,
            num_parallel_trials,
        ),
    )
    worker.start()
    return worker
