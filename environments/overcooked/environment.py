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

import logging
import os

import cogment
import gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS, Overcooked, OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from cogment_verse.constants import PLAYER_ACTOR_CLASS
from cogment_verse.specs import EnvironmentSpecs

# configure pygame to use a dummy video server to be able to render headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

log = logging.getLogger(__name__)


class Environment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.impl_name = cfg.env_name

        base_mdp = OvercookedGridworld.from_layout_name(self.cfg.layout)
        env = OvercookedEnv.from_mdp(base_mdp, **DEFAULT_ENV_PARAMS)
        gym_env = Overcooked(base_env=env, featurize_fn=env.featurize_state_mdp, baselines_reproducible=True)

        self.env_specs = EnvironmentSpecs.create_homogeneous(
            num_players=self.cfg.num_players,
            turn_based=self.cfg.turn_based,
            observation_space=gym_env.observation_space,
            action_space=gym_env.action_space,
            web_components_file="OvercookedRealTime.js" if self.impl_name.endswith("real-time") else "OvercookedTurnBased.js",
        )

    def get_web_components_dir(self):
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), "web", "dist")

    def get_implementation_name(self):
        return self.impl_name

    def get_environment_specs(self):
        return self.env_specs

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        non_player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name != PLAYER_ACTOR_CLASS
        ]

        session_cfg = environment_session.config

        base_mdp = OvercookedGridworld.from_layout_name(self.cfg.layout)
        env = OvercookedEnv.from_mdp(base_mdp, **DEFAULT_ENV_PARAMS)
        gym_env = Overcooked(base_env=env, featurize_fn=env.featurize_state_mdp, baselines_reproducible=True)
        gym_observation = gym_env.reset()["both_agent_obs"]

        observation_space = self.env_specs.get_observation_space(session_cfg.render_width)
        action_space = self.env_specs.get_action_space()

        observations = []
        for player_actor_idx, player_actor_name in player_actors:
            observation = observation_space.create(
                value=gym_observation[player_actor_idx],
                rendered_frame=gym_env.render() if session_cfg.render else None,
                overridden_players=[],
            )
            observations.append((player_actor_name, observation_space.serialize(observation)))

        for _, player_actor_name in non_player_actors:
            observation = observation_space.create(
                value=gym_observation[0],  # Dummy observation for non-player actors
                rendered_frame=gym_env.render() if session_cfg.render else None,
                overridden_players=[],
            )
            observations.append((player_actor_name, observation_space.serialize(observation)))

        environment_session.start(observations)
        async for event in environment_session.all_events():
            if event.actions:

                joint_action = []
                for player_actor_idx, player_actor_name in player_actors:
                    player_action = action_space.deserialize(
                        event.actions[player_actor_idx].action,
                    )
                    action_value = player_action.value

                    # Clipped action and send to gym environment
                    if isinstance(gym_env.action_space, gym.spaces.Box):
                        action_value = np.clip(action_value, gym_env.action_space.low, gym_env.action_space.high)

                    joint_action.append(action_value)

                gym_observation, reward, done, _info = gym_env.step((joint_action))

                observations = []
                for player_actor_idx, player_actor_name in player_actors:
                    observation = observation_space.create(
                        value=gym_observation["both_agent_obs"][player_actor_idx],
                        rendered_frame=gym_env.render() if session_cfg.render else None,
                        overridden_players=[],
                    )
                    observations.append((player_actor_name, observation_space.serialize(observation)))

                for _, player_actor_name in non_player_actors:
                    observation = observation_space.create(
                        value=gym_observation["both_agent_obs"][0],  # Dummy observation for non-player actors
                        rendered_frame=gym_env.render() if session_cfg.render else None,
                        overridden_players=[],
                    )
                    observations.append((player_actor_name, observation_space.serialize(observation)))

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
