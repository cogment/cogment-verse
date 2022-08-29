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

import os

import cogment
from cogment.environment import EnvironmentSession
import gym
import numpy as np

from cogment_verse.specs import (
    EnvironmentSpecs,
    Observation,
    encode_rendered_frame,
    gym_action_from_action,
    observation_from_gym_observation,
    space_from_gym_space,
)

from cogment_verse.constants import PLAYER_ACTOR_CLASS
from debug.mp_pdb import ForkedPdb

# configure pygame to use a dummy video server to be able to render headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"


# pylint: disable=no-member
class Environment:
    """openGym environement"""

    def __init__(self, cfg: dict):
        self.gym_env_name = cfg.env_name
        self.gym_env = gym.make(self.gym_env_name)
        self.env_specs = EnvironmentSpecs(
            num_players=1,
            turn_based=False,
            observation_space=space_from_gym_space(self.gym_env.observation_space),
            action_space=space_from_gym_space(self.gym_env.action_space),
        )

    def get_implementation_name(self):
        """Environment name"""
        return self.gym_env_name

    def get_environment_specs(self):
        """Environment specs"""
        return self.env_specs

    async def impl(self, environment_session: EnvironmentSession):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        # assert len(player_actors) == 1
        [(player_actor_idx, player_actor_name)] = player_actors

        # Reset environment
        session_cfg = environment_session.config
        gym_observation, _ = self.gym_env.reset(seed=session_cfg.seed, return_info=True)
        observation_value = observation_from_gym_observation(self.gym_env.observation_space, gym_observation)

        rendered_frame = None
        if session_cfg:
            rendered_frame = encode_rendered_frame(self.gym_env.render(mode="rgb_array"), session_cfg.render_width)

        environment_session.start([("*", Observation(value=observation_value, rendered_frame=rendered_frame))])
        async for event in environment_session.all_events():
            if event.actions:
                # Get action from actor through orchestrator
                action_value = event.actions[player_actor_idx].action.value
                gym_action = gym_action_from_action(self.env_specs.action_space, action_value)

                # Clipped action and send to gym environment
                if isinstance(self.env_specs.action_space, gym.spaces.Box):
                    clipped_action = np.clip(gym_action, self.gym_env.action_space.low, self.gym_env.action_space.high)
                else:
                    clipped_action = gym_action
                gym_observation, reward, done, _ = self.gym_env.step(clipped_action)
                observation_value = observation_from_gym_observation(self.gym_env.observation_space, gym_observation)

                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = encode_rendered_frame(
                        self.gym_env.render(mode="rgb_array"), session_cfg.render_width
                    )

                # Encode observation grpc format
                observations = [
                    (
                        "*",
                        Observation(value=observation_value, rendered_frame=rendered_frame, overridden_players=[]),
                    ),
                ]

                # Send reward & observation to orchestrator
                if reward is not None:
                    environment_session.add_reward(value=reward, confidence=1.0, to=[player_actor_name])

                if done:
                    environment_session.end(observations)
                elif event.type != cogment.EventType.ACTIVE:
                    environment_session.end(observations)
                else:
                    environment_session.produce_observations(observations)

        self.gym_env.close()
