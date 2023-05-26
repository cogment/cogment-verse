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

import cogment

import torch
from cogment_verse.constants import ActorClass
from cogment_verse.specs.environment_specs import EnvironmentSpecs

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

log = logging.getLogger(__name__)

# pylint: disable=arguments-differ
class SB3PPOActor:
    def __init__(self, _cfg):
        super().__init__()
        self._dtype = torch.float
        self.cfg = _cfg
        print(f"SB3Actor config: {_cfg}")

    def get_actor_classes(self):
        return [ActorClass.PLAYER.value]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config
        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        observation_space = environment_specs.get_observation_space()
        action_space = environment_specs.get_action_space(seed=config.seed)

        # print(f"SB3Actor agent_config: {actor_session.config}")

        # Get model
        checkpoint = load_from_hub(
            repo_id=actor_session.config.hf_hub_model.repo_id,
            filename=actor_session.config.hf_hub_model.filename,
        )

        model = PPO.load(checkpoint)
        model.policy.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation = observation_space.deserialize(event.observation.observation)
                if observation.current_player is not None and observation.current_player != actor_session.name:
                    # Not the turn of the agent
                    actor_session.do_action(action_space.serialize(action_space.create()))
                    continue

                observation_tensor = torch.unsqueeze(torch.tensor(observation.value, dtype=self._dtype)[:, :, :4].permute(2, 0, 1), dim=0) #.view(1, -1)

                # Get action from policy network
                with torch.no_grad():
                    actions, dist, log_probs = model.policy(observation_tensor)
                    #dist = model.policy(observation_tensor)

                    action_value = actions.cpu().numpy()[0]

                # Send action to environment
                action = action_space.create(value=action_value)
                actor_session.do_action(action_space.serialize(action))
