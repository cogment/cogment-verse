# Copyright 2020 Artificial Intelligence Redefined <dev+cogment@ai-r.com>
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
import functools
from typing import List
from cogment.control import TrialState
from cogment_verse.specs import cog_settings
import cogment


TRIAL_DATASTORE_ENDPOINT = "grpc://localhost:9001"


def make_sync(func):
    """Simple wrapper function that runs an async function using asyncio making it appear sync"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


async def get_trial_ids(endpoint: str, user_id: str = "thunderblade_client"):
    trial_ids = []
    cog_context = cogment.Context(cog_settings=cog_settings, user_id=user_id)
    datastore = cog_context.get_datastore(endpoint=cogment.Endpoint(TRIAL_DATASTORE_ENDPOINT))

    async for trial in datastore.all_trials(bundle_size=50):
        if trial.trial_state == TrialState.ENDED:
            trial_ids.append(trial.trial_id)

    return trial_ids


async def get_trial_data(endpoint: str, trial_ids: List[str] = [], user_id: str = "thunderblade_client"):
    """Extracts the trial data for the specified trial_ids.

    Returns a dictionary with keys as trial ids.
    args:
        trial_ids: If None, will use all available trial ids in the datastore for extraction.
    """
    cog_context = cogment.Context(cog_settings=cog_settings, user_id=user_id)
    datastore = cog_context.get_datastore(endpoint=cogment.Endpoint(TRIAL_DATASTORE_ENDPOINT))

    if not trial_ids:
        async for trial in datastore.all_trials(bundle_size=10):
            if trial.trial_state == TrialState.ENDED:
                trial_ids.append(trial.trial_id)

    trial_data = {}

    for trial_id in trial_ids:
        trial_data[trial_id] = {}

        trial_info = await datastore.get_trials([trial_id])
        trial_data[trial_id]["trial_info"] = trial_info[0]

        samples = []
        async for sample in datastore.all_samples(trial_info):
            samples.append(sample)

        trial_data[trial_id]["samples"] = samples

    return trial_data
