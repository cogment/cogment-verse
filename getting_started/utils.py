# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import cogment
from cogment.control import TrialState
import numpy as np

from cogment_verse.specs import cog_settings
from cogment_verse.specs.environment_specs import EnvironmentSpecs


def make_sync(func):
    """Simple wrapper function that runs an async function using asyncio making it appear sync"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


async def get_trial_ids(endpoint: str, user_id: str = "cogment_verse_client"):
    trial_ids = []
    cog_context = cogment.Context(cog_settings=cog_settings, user_id=user_id)
    datastore = cog_context.get_datastore(endpoint=cogment.Endpoint(endpoint))

    async for trial in datastore.all_trials(bundle_size=50):
        if trial.trial_state == TrialState.ENDED:
            trial_ids.append(trial.trial_id)

    return trial_ids


async def get_trial_data(endpoint: str, trial_ids: List[str], user_id: str = "cogment_verse_client"):
    """Extracts the trial data for the specified trial_ids.

    Returns a dictionary with keys as trial ids.
    args:
        trial_ids: If None, will use all available trial ids in the datastore for extraction.
    """
    cog_context = cogment.Context(cog_settings=cog_settings, user_id=user_id)
    datastore = cog_context.get_datastore(endpoint=cogment.Endpoint(endpoint))

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


async def get_gym_trial_data(endpoint: str, trial_ids: List[str], actor_names, user_id: str = "cogment_verse_client"):
    """Extracts the trial data for the specified trial_ids and actor names. In addition, the observations and actions will be
    deserialized in the proper gym space format.

    Returns a lsit of tuples with the following elements:
        (trial_id, tick_id, trial_state, actor_name, gym_observation, gym_action, reward)
    args:
        trial_ids: If None, will use all available trial ids in the datastore for extraction.
        actor_names: If None, will return for all available actors in the sample.
    """
    trial_data = await get_trial_data(endpoint, trial_ids, user_id)

    gym_samples = []
    for _, data in trial_data.items():
        trial_info = data["trial_info"]
        environment_specs = EnvironmentSpecs.deserialize(trial_info.parameters.actors[0].config.environment_specs)
        action_space = environment_specs.get_action_space()
        observation_space = environment_specs.get_observation_space()

        for sample in data["samples"]:
            for actor_name, actor_data in sample.actors_data.items():
                if actor_names and actor_name not in actor_names:
                    continue

                gym_observation = observation_space.deserialize(actor_data.observation).value
                gym_action = action_space.deserialize(actor_data.action).value
                reward = actor_data.reward
                gym_samples.append(
                    (
                        sample.trial_id,
                        sample.tick_id,
                        sample.trial_state,
                        actor_name,
                        gym_observation,
                        gym_action,
                        reward,
                    )
                )

    return gym_samples


def array_equal(array_1: np.ndarray, array_2: np.ndarray, tolerance=1e-05) -> bool:
    """Returns equality of two arrays, based on element-wise equality with a tolerance for close values.."""
    if np.allclose(array_1, array_2, atol=tolerance):
        return True
    return False


def find_array_index(arr_list: List[np.ndarray], test_array: np.ndarray) -> int:
    """Find the index of a test array in a list of arrays, with a tolerance for equality."""
    for i, arr in enumerate(arr_list):
        if array_equal(arr, test_array):
            return i
    return -1  # Test array not found in the list


def count_unique_arrays(arr_list: List[np.ndarray], compare_func=array_equal) -> int:
    """Count unique arrays within a list of arrays."""
    unique_arrays = []
    for arr in arr_list:
        is_unique = True
        for unique_arr in unique_arrays:
            if compare_func(arr, unique_arr):
                is_unique = False
                break
        if is_unique:
            unique_arrays.append(arr)
    return len(unique_arrays)
