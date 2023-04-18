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

import logging
import os
import time

import cogment
import gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from overcooked_ai_py.mdp.overcooked_env import Overcooked, OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from cogment_verse.constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS
from cogment_verse.specs import EnvironmentSpecs

# configure pygame to use a dummy video server to be able to render headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

log = logging.getLogger(__name__)


class OvercookedEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gym_env_name = cfg.env_name

        base_mdp = OvercookedGridworld.from_layout_name(self.cfg.layout)
        env = OvercookedEnv.from_mdp(base_mdp, **DEFAULT_ENV_PARAMS)
        gym_env = Overcooked(base_env=env, featurize_fn=env.featurize_state_mdp)

        self.env_specs = EnvironmentSpecs.create_homogeneous(
            num_players=2,
            turn_based=False,
            observation_space=gym_env.observation_space,
            action_space=gym_env.action_space,
        )

    def get_implementation_name(self):
        return self.gym_env_name

    def get_environment_specs(self):
        return self.env_specs

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]

        session_cfg = environment_session.config

        base_mdp = OvercookedGridworld.from_layout_name(self.cfg.layout)
        env = OvercookedEnv.from_mdp(base_mdp, **DEFAULT_ENV_PARAMS)
        gym_env = Overcooked(base_env=env, featurize_fn=env.featurize_state_mdp)

        observation_space = self.env_specs.get_observation_space(session_cfg.render_width)
        action_space = self.env_specs.get_action_space()

        gym_observation = gym_env.reset()["both_agent_obs"]

        observation = observation_space.create(
            value=gym_observation,
            rendered_frame=gym_env.render() if session_cfg.render else None,
        )

        environment_session.start([("*", observation_space.serialize(observation))])
        async for event in environment_session.all_events():
            if event.actions:

                # print(f"players in env event: {len(event.actions)}")
                print(f"env agent_idx: {gym_env.agent_idx}")
                print(f"player_idx: {0} | action: {action_space.deserialize(event.actions[0].action).value}")
                print(f"player_idx: {1} | action: {action_space.deserialize(event.actions[1].action).value}")

                joint_action = []
                for player_actor_idx, player_actor_name in player_actors:
                    player_action = action_space.deserialize(
                        event.actions[player_actor_idx].action,
                    )
                    action_value = player_action.value

                    # Clipped action and send to gym environment
                    if isinstance(gym_env.action_space, gym.spaces.Box):
                        action_value = np.clip(action_value, gym_env.action_space.low, gym_env.action_space.high)

                    # print(f"player_action: {player_action}")
                    # print(f"ACTION: {action_value}")

                    joint_action.append(action_value)

                gym_observation, reward, done, _info = gym_env.step((joint_action))

                observation = observation_space.create(
                    value=gym_observation["both_agent_obs"],
                    rendered_frame=gym_env.render() if session_cfg.render else None,
                    overridden_players=[],
                )

                observations = [("*", observation_space.serialize(observation))]

                if reward is not None:
                    environment_session.add_reward(
                        value=reward,
                        confidence=1.0,
                        to=[player_actor_name],
                    )

                if done:
                    # The trial ended
                    environment_session.end(observations)
                elif event.type != cogment.EventType.ACTIVE:
                    # The trial termination has been requested
                    environment_session.end(observations)
                else:
                    # The trial is active
                    environment_session.produce_observations(observations)

        gym_env.close()
