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
from enum import Enum
from typing import Tuple

import cogment
import gymnasium as gymna
import numpy as np
import supersuit as ss

from cogment_verse.constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS
from cogment_verse.specs import (
    EnvironmentSpecs,
    Observation,
    SpaceMask,
    encode_rendered_frame,
    gym_action_from_action,
    observation_from_gym_observation,
    space_from_gym_space,
)
from cogment_verse.utils import import_class

log = logging.getLogger(__name__)


def action_mask_from_pz_action_mask(pz_action_mask):
    return SpaceMask(
        properties=[
            SpaceMask.PropertyMask(
                discrete=[action_idx for action_idx, action in enumerate(pz_action_mask) if action == 1]
            )
        ]
    )


def get_pz_player(observation: np.ndarray, actor_names: list) -> Tuple[str, int]:
    """Get name and index for the petting zoo player. Note that it works specifically to Atari game"""
    num_agents = len(actor_names)
    indicators = observation[0, 0, -num_agents:]
    idx = int(np.where(indicators)[0])
    current_agent_name = actor_names[idx]

    return current_agent_name, idx


def get_rl_agent(current_pz_agent_name: str, actor_names: list) -> Tuple[str, int]:
    """Get index and name for reinforcement leanring"""
    if len(actor_names) == 1:
        return (actor_names[0], 0)
    idx = [count for (count, agent_name) in enumerate(actor_names) if agent_name == current_pz_agent_name][0]
    return (actor_names[idx], idx)


def atari_env_wrapper(env: gymna.Env) -> gymna.Env:
    """Wrapper for atari env"""
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)

    return env


class PzEnvType(Enum):
    CLASSIC = "classic"
    ATARI = "atari"


class Environment:
    def __init__(self, cfg):
        self.env_class_name = cfg.env_class_name
        self.env_type_str = self.env_class_name.split(".")[1]
        if self.env_type_str not in [pz_env_type.value for pz_env_type in PzEnvType]:
            raise RuntimeError(f"PettingZoo adapter does not support environments of type [{self.env_type_str}]")

        self.env_type = PzEnvType(self.env_type_str)
        self.env_class = import_class(self.env_class_name)
        pz_env = self.env_class.env()
        if self.env_type_str == "atari":
            pz_env = atari_env_wrapper(pz_env)

        num_players = 0
        observation_space = None
        action_space = None
        for player in pz_env.possible_agents:
            num_players += 1
            if observation_space is None:
                observation_space = pz_env.observation_space(player)
                action_space = pz_env.action_space(player)
            else:
                if observation_space != pz_env.observation_space(player) or action_space != pz_env.action_space(player):
                    raise RuntimeError(
                        "Petting zoo environment with heterogeneous action/observation spaces are not supported yet"
                    )

        assert num_players >= 1
        self.env_specs = EnvironmentSpecs(
            num_players=num_players,
            observation_space=space_from_gym_space(observation_space),
            action_space=space_from_gym_space(action_space),
            turn_based=self.env_type in [PzEnvType.CLASSIC],
        )

    def get_implementation_name(self):
        return self.env_class_name

    def get_environment_specs(self):
        return self.env_specs

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        actor_names = [actor.actor_name for actor in actors if actor.actor_class_name == PLAYER_ACTOR_CLASS]
        session_cfg = environment_session.config

        # Reset environment.
        if session_cfg.render:
            pz_env = self.env_class.env(render_mode="rgb_array")
        else:
            pz_env = self.env_class.env()
        if self.env_type_str == "atari":
            pz_env = atari_env_wrapper(pz_env)
        pz_env.reset(seed=session_cfg.seed)
        pz_observation, _, _, _, _ = pz_env.last()

        if len(pz_env.agents) != len(actor_names) and len(actor_names) > 1:
            raise ValueError(f"Number of actors does not match environments requirement ({len(pz_env.agents)} actors)")

        pz_player_name, _ = get_pz_player(observation=pz_observation, actor_names=pz_env.agents)
        rl_actor_name, rl_actor_idx = get_rl_agent(current_pz_agent_name=pz_player_name, actor_names=actor_names)
        observation_value = observation_from_gym_observation(pz_env.observation_space(pz_player_name), pz_observation)

        # Render the pixel for UI
        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in pz_env.metadata["render_modes"]:
                log.warning(f"Petting Zoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            rendered_frame = encode_rendered_frame(pz_env.render(), session_cfg.render_width)

        environment_session.start(
            [
                (
                    "*",
                    Observation(
                        value=observation_value,  # TODO Should only be sent to the current player
                        rendered_frame=rendered_frame,  # TODO Should only be sent to observers
                        current_player=rl_actor_name,
                    ),
                )
            ]
        )
        async for event in environment_session.all_events():
            if event.actions:
                # Action
                player_action_value = event.actions[rl_actor_idx].action.value
                action_value = player_action_value

                # Observation
                gym_action = gym_action_from_action(
                    self.env_specs.action_space, action_value  # pylint: disable=no-member
                )
                pz_env.step(gym_action)
                pz_observation, pz_reward, done, _, _ = pz_env.last()

                # Actor names
                pz_player_name, _ = get_pz_player(observation=pz_observation, actor_names=pz_env.agents)
                rl_actor_name, rl_actor_idx = get_rl_agent(
                    current_pz_agent_name=pz_player_name, actor_names=actor_names
                )

                # Send data
                observation_value = observation_from_gym_observation(
                    pz_env.observation_space(pz_player_name), pz_observation
                )

                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = encode_rendered_frame(pz_env.render(), session_cfg.render_width)
                observations = [
                    (
                        "*",
                        Observation(
                            value=observation_value,
                            rendered_frame=rendered_frame,
                            current_player=rl_actor_name,
                        ),
                    )
                ]
                environment_session.add_reward(
                    value=pz_reward,
                    confidence=1.0,
                    to=[rl_actor_name],
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

        pz_env.close()
