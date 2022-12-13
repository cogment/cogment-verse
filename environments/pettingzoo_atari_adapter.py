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

from typing import Tuple
from enum import Enum
import logging

import cogment

from cogment_verse.specs import (
    encode_rendered_frame,
    EnvironmentSpecs,
    Observation,
    SpaceMask,
    space_from_gym_space,
    gym_action_from_action,
    observation_from_gym_observation,
)
from cogment_verse.constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS
from cogment_verse.utils import import_class
import supersuit as ss
import numpy as np
from debug.mp_pdb import ForkedPdb

log = logging.getLogger(__name__)


def action_mask_from_pz_action_mask(pz_action_mask):
    return SpaceMask(
        properties=[
            SpaceMask.PropertyMask(
                discrete=[action_idx for action_idx, action in enumerate(pz_action_mask) if action == 1]
            )
        ]
    )


def get_current_agent(observation: np.ndarray, agent_names: list) -> Tuple[str, int]:
    """Get name and index for the current agent. Note that it works specific to Atari game"""
    num_agents = len(agent_names)
    indicators = observation[0, 0, -num_agents:]
    idx = int(np.where(indicators)[0])
    current_agent_name = agent_names[idx]

    return current_agent_name, idx


class PzEnvType(Enum):
    CLASSIC = "classic"
    ATARI = "atari"


class Environment:
    def __init__(self, cfg):
        self.env_class_name = cfg.env_class_name
        env_type_str = self.env_class_name.split(".")[1]
        if env_type_str not in [pz_env_type.value for pz_env_type in PzEnvType]:
            raise RuntimeError(f"PettingZoo adapter does not support environments of type [{env_type_str}]")
        self.env_type = PzEnvType(env_type_str)
        self.env_class = import_class(self.env_class_name)

        self.pz_env = self.env_class.env(render_mode="rgb_array")

        if env_type_str == "atari":
            self.pz_env = ss.max_observation_v0(self.pz_env, 2)
            self.pz_env = ss.frame_skip_v0(self.pz_env, 4)
            self.pz_env = ss.clip_reward_v0(self.pz_env, lower_bound=-1, upper_bound=1)
            self.pz_env = ss.color_reduction_v0(self.pz_env, mode="B")
            self.pz_env = ss.resize_v1(self.pz_env, x_size=84, y_size=84)
            self.pz_env = ss.frame_stack_v1(self.pz_env, 4)
            self.pz_env = ss.agent_indicator_v0(self.pz_env, type_only=False)

        num_players = 0
        observation_space = None
        action_space = None

        for player in self.pz_env.possible_agents:
            num_players += 1
            if observation_space is None:
                observation_space = self.pz_env.observation_space(player)
                action_space = self.pz_env.action_space(player)
            else:
                if observation_space != self.pz_env.observation_space(
                    player
                ) or action_space != self.pz_env.action_space(player):
                    raise RuntimeError(
                        "Petting zoo environment with heterogeneous action/observation spaces are not supported yet"
                    )

        assert num_players >= 1
        self.env_specs = EnvironmentSpecs(
            num_players=num_players,
            observation_space=space_from_gym_space(observation_space),
            action_space=space_from_gym_space(action_space),
            turn_based=self.env_type in [PzEnvType.CLASSIC, PzEnvType.ATARI],
        )

    def get_implementation_name(self):
        return self.env_class_name

    def get_environment_specs(self):
        return self.env_specs

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        (current_player_actor_idx, current_player_actor_name) = player_actors[0]

        # No support for teachers
        teacher_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == TEACHER_ACTOR_CLASS
        ]
        assert len(teacher_actors) == 0

        session_cfg = environment_session.config
        self.pz_env.reset(seed=session_cfg.seed)
        pz_observation, _pz_reward, _pz_done, _pz_trunc, _pz_info = self.pz_env.last()
        current_player_pz_agent, _ = get_current_agent(observation=pz_observation, agent_names=self.pz_env.agents)

        observation_value = observation_from_gym_observation(
            self.pz_env.observation_space(current_player_pz_agent), pz_observation
        )

        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in self.pz_env.metadata["render_modes"]:
                log.warning(f"Petting Zoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            self.pz_env.render_mode = "rgb_array"
            rendered_frame = encode_rendered_frame(self.pz_env.render(), session_cfg.render_width)

        environment_session.start(
            [
                (
                    "*",
                    Observation(
                        value=observation_value,  # TODO Should only be sent to the current player
                        rendered_frame=rendered_frame,  # TODO Should only be sent to observers
                        current_player=current_player_actor_name,
                    ),
                )
            ]
        )
        async for event in environment_session.all_events():
            if event.actions:
                player_action_value = event.actions[current_player_actor_idx].action.value
                action_value = player_action_value
                gym_action = gym_action_from_action(
                    self.env_specs.action_space, action_value  # pylint: disable=no-member
                )
                self.pz_env.step(gym_action)
                pz_observation, pz_reward, done, py_trunc, pz_info = self.pz_env.last()
                current_player_pz_agent, _ = get_current_agent(
                    observation=pz_observation, agent_names=self.pz_env.agents
                )
                observation_value = observation_from_gym_observation(
                    self.pz_env.observation_space(current_player_pz_agent), pz_observation
                )

                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = encode_rendered_frame(
                        self.pz_env.render(), session_cfg.render_width
                    )

                observations = [
                    (
                        "*",
                        Observation(
                            value=observation_value,
                            rendered_frame=rendered_frame,
                            current_player=current_player_actor_name,
                        ),
                    )
                ]

                # for (rewarded_player_pz_agent, pz_reward) in self.pz_env.rewards.items():
                #     if pz_reward == 0:
                #         continue
                #     rewarded_player_actor_name = next(
                #         player_actor_name
                #         for (player_pz_agent, (player_actor_idx, player_actor_name)) in zip(
                #             self.pz_env.agents, player_actors
                #         )
                #         if player_pz_agent == rewarded_player_pz_agent
                #     )
                environment_session.add_reward(
                    value=pz_reward,
                    confidence=1.0,
                    to=[current_player_actor_name],
                )
                # done = all(self.pz_env.dones[pz_agent] for pz_agent in self.pz_env.agents)

                if done:
                    # The trial ended
                    environment_session.end(observations)
                elif event.type != cogment.EventType.ACTIVE:
                    # The trial termination has been requested
                    environment_session.end(observations)
                else:
                    # The trial is active
                    environment_session.produce_observations(observations)

        self.pz_env.close()
