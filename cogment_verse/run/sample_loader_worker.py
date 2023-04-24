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
from cogment.control import TrialState

from ..services_directory import ServiceType

log = logging.getLogger(__name__)


class SampleQueueEvent:
    def __init__(self, trial_id=None, trial_idx=None, sample=None, done=False):
        self.trial_id = trial_id
        self.trial_idx = trial_idx
        self.sample = sample
        self.done = done


class SampleloaderSession:
    def __init__(self, datastore, trial_idx, trial_info, sample_queue, impl):
        self.trial_idx = trial_idx
        self.datastore = datastore
        self.trial_info = trial_info
        self.sample_queue = sample_queue
        self.impl = impl

    def produce_sample(self, sample):
        self.sample_queue.put(
            SampleQueueEvent(trial_id=self.trial_info.trial_id, trial_idx=self.trial_idx, sample=sample)
        )

    def all_trial_samples(self):
        return self.datastore.all_samples([self.trial_info])

    def create_task(self):
        async def wrapped_impl():
            try:
                await self.impl(self)
            except KeyboardInterrupt:
                # This one is ignored, it's logged at a bunch of different places
                pass
            except Exception as error:
                log.error(
                    f"Uncaught error occured during the sample production for trial [{self.trial_info.trial_id}]",
                    exc_info=error,
                )
                raise

        return asyncio.create_task(wrapped_impl())


async def async_sample_loader_worker(trial_ids, num_trials, sample_queue, impl, services_directory):

    # Importing 'specs' only in the subprocess (i.e. where generate has been properly executed)
    # pylint: disable-next=import-outside-toplevel
    from cogment_verse.specs import cog_settings

    cog_context = cogment.Context(cog_settings=cog_settings, user_id="cogment_verse_sample_loader")
    datastore = cog_context.get_datastore(
        endpoint=cogment.Endpoint(services_directory.get(ServiceType.TRIAL_DATASTORE))
    )
    # Define a timeout for trial info retrieval
    # pylint: disable-next=protected-access
    datastore._timeout = 100

    if not trial_ids:
        async for trial in datastore.all_trials(bundle_size=10):
            if trial.trial_state == TrialState.ENDED:
                trial_ids.append(trial.trial_id)

    if num_trials > -1:
        trial_ids = trial_ids[:num_trials]

    log.warning(f"Number of trials retrieved from datastore: {len(trial_ids)}")

    sample_loader_tasks = []
    trial_idx = 0
    for trial_id in trial_ids:

        trials_info = await datastore.get_trials([trial_id])
        if len(trials_info) == 0:
            log.warning(f"Trial [{trial_id}] couldn't be found in the trial datastore, retrying later")
            continue

        [trial_info] = trials_info

        log.debug(f"[{trial_info.trial_id}] fetched from datastore.")

        sample_loader_session = SampleloaderSession(datastore, trial_idx, trial_info, sample_queue, impl)

        sample_loader_tasks.append(sample_loader_session.create_task())
        trial_idx += 1

    if len(sample_loader_tasks) > 0:
        await asyncio.wait(sample_loader_tasks)

    sample_queue.put(SampleQueueEvent(done=True))


def sample_loader_worker(trial_ids, num_trials, sample_queue, impl, services_directory):

    try:
        asyncio.run(async_sample_loader_worker(trial_ids, num_trials, sample_queue, impl, services_directory))
    except KeyboardInterrupt:
        sys.exit(-1)


def start_sample_loader_worker(trial_ids, num_trials, sample_queue, impl, services_directory):

    worker = Process(
        name="sample_loader_worker",
        target=sample_loader_worker,
        args=(
            trial_ids,
            num_trials,
            sample_queue,
            impl,
            services_directory,
        ),
    )
    worker.start()
    return worker
