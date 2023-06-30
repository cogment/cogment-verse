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
import logging
from typing import Awaitable, Callable
from multiprocessing import Queue
from typing import Any

from cogment.datastore import Datastore, DatastoreSample
from cogment.model_registry_v2 import ModelRegistry
from cogment.session import ActorInfo

from .environment_specs import EnvironmentSpecs
from .session_helper import SessionHelper

log = logging.getLogger(__name__)


class SampleQueueEvent:
    def __init__(self, trial_id=None, trial_idx=None, sample=None, done=False):
        self.trial_id = trial_id
        self.trial_idx = trial_idx
        self.sample = sample
        self.done = done


class SampleProducerSession(SessionHelper):
    def __init__(
        self,
        datastore: Datastore,
        trial_idx: int,
        trial_info,
        sample_queue: Queue,
        model_registry: ModelRegistry,
        impl: Callable[["SampleProducerSession"], Awaitable],
    ):
        self.trial_idx = trial_idx
        self.datastore = datastore
        self.trial_info = trial_info
        self.sample_queue = sample_queue
        self.model_registry = model_registry
        self.impl = impl

        actor_infos = []
        environment_specs = None
        for actor_params in self.trial_info.parameters.actors:
            actor_infos.append(ActorInfo(name=actor_params.name, class_name=actor_params.class_name))
            if environment_specs is None:
                environment_specs = EnvironmentSpecs.deserialize(actor_params.config.environment_specs)
        super().__init__(
            actor_infos=actor_infos,
            environment_specs=environment_specs,
        )

    def produce_sample(self, sample):
        self.sample_queue.put(
            SampleQueueEvent(trial_id=self.trial_info.trial_id, trial_idx=self.trial_idx, sample=sample)
        )

    def all_trial_samples(self) -> DatastoreSample:
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

    def get_observation(self, tick_data: Any, actor_name: str):
        # For sample producers, tick_datas are samples
        sample = tick_data

        observation_space = self.get_observation_space(actor_name)

        return observation_space.deserialize(sample.actors_data[actor_name].observation)

    def get_player_observations(self, tick_data: Any, actor_name: str = None):
        if actor_name is None:
            observations = [self.get_observation(tick_data, actor_name) for player_actor_name in self.player_actors]
            if len(observations) == 0:
                raise RuntimeError("No player actors")
            if len(observations) > 1:
                raise RuntimeError("More than 1 player actor, please provide an actor name")
            return observations[0]

        observations = [
            self.get_observation(tick_data, actor_name)
            for player_actor_name in self.player_actors
            if player_actor_name == actor_name
        ]
        if len(observations) == 0:
            raise RuntimeError(f"No player actors having name [{actor_name}]")
        return observations[0]

    def get_action(self, tick_data: Any, actor_name: str):
        # For sample producers, tick_datas are samples
        sample = tick_data

        action_space = self.get_action_space(actor_name)
        return action_space.deserialize(sample.actors_data[actor_name].action)

    def get_reward(self, tick_data: Any, actor_name: str):
        # For sample producers, tick_datas are samples
        sample = tick_data

        return sample.actors_data[actor_name].reward
