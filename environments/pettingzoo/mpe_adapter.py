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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import cogment
import gymnasium as gymna
import numpy as np
import supersuit as ss
from cogment.environment import EnvironmentSession

from cogment_verse.constants import EVALUATOR_ACTOR_CLASS, PLAYER_ACTOR_CLASS, WEB_ACTOR_NAME
from cogment_verse.specs import ActorSpecs
from cogment_verse.specs.ndarray_serialization import SerializationFormat, deserialize_ndarray
from cogment_verse.utils import import_class
from environments.pettingzoo.utils import PettingZooEnvType

log = logging.getLogger(__name__)


# TODO: Move to utils
def is_homogeneous(env) -> bool:
    num_players = 0
    observation_space = None
    action_space = None
    for player in env.possible_agents:
        num_players += 1
        print(f"player | observation_space: {env.observation_space(player)} | action_space: {env.action_space(player)}")
        if observation_space is None:
            observation_space = env.observation_space(player)
            action_space = env.action_space(player)
        else:
            if observation_space != env.observation_space(player) or action_space != env.action_space(player):
                return False
    return True


def get_player(observation: np.ndarray, actor_names: list) -> Tuple[str, int]:
    """Get name and index for the PettingZoo player. Note that it works specifically to Atari game"""
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


class MpeEnvironment:
    """MPE PettingZoo e.g., Simple Adversary. Simple Tag, etc."""

    def __init__(self, cfg):
        self.env_cfg = cfg
        self.env_class_name = cfg.env_class_name
        self.env_type_str = self.env_class_name.split(".")[1]
        if self.env_type_str not in [env_type.value for env_type in PettingZooEnvType]:
            raise RuntimeError(f"PettingZoo adapter does not support environments of type [{self.env_type_str}]")

        self.env_type = PettingZooEnvType(self.env_type_str)
        assert self.env_type == PettingZooEnvType.MPE
        self.env_class = import_class(self.env_class_name)
        # TODO: if class is mpe, then env = ()

        print(f"env cfg: {cfg}")

        env = self.env_class.env(
            num_good=self.env_cfg.num_good,
            num_adversaries=self.env_cfg.num_adversaries,
            num_obstacles=self.env_cfg.num_obstacles,
            max_cycles=self.env_cfg.max_cycles,
            continuous_actions=self.env_cfg.continuous_actions
        )

        print(f"env.agents: {env.possible_agents}")

        if not is_homogeneous(env):
            raise RuntimeError(
                "PettingZoo environment with heterogeneous action/observation spaces are not supported yet"
            )

        assert len(env.possible_agents) >= 1
        self.env_specs = ActorSpecs.create_homogeneous(
            num_players=len(env.possible_agents),
            observation_space=env.observation_space(env.possible_agents[0]),
            action_space=env.action_space(env.possible_agents[0]),
            turn_based=False,
        )

    def get_implementation_name(self):
        return self.env_class_name

    def get_environment_specs(self):
        return self.env_specs

    @abstractmethod
    async def impl(self, environment_session: EnvironmentSession):
        raise NotImplementedError

    async def impl(self, environment_session: EnvironmentSession):
        actors = environment_session.get_active_actors()
        print(f"env service actors: {actors}")
        actor_names = [actor.actor_name for actor in actors if actor.actor_class_name == PLAYER_ACTOR_CLASS]
        web_actor_idx = [count for (count, actor_name) in enumerate(actor_names) if actor_name == WEB_ACTOR_NAME]
        session_cfg = environment_session.config

        # Initialize environment
        #env = self.env_class.env() if session_cfg.render else self.env_class.env()
        env = self.env_class.env(
            num_good=self.env_cfg.num_good,
            num_adversaries=self.env_cfg.num_adversaries,
            num_obstacles=self.env_cfg.num_obstacles,
            max_cycles=self.env_cfg.max_cycles,
            continuous_actions=self.env_cfg.continuous_actions,
            render_mode="rgb_array" if session_cfg.render else None,
        )
        observation_space = self.env_specs.get_observation_space(session_cfg.render_width)
        action_space = self.env_specs.get_action_space()

        # Reset environment
        env.reset(seed=session_cfg.seed)
        agent_iter = iter(env.agent_iter())
        pz_observation, _, _, _, _ = env.last()

        print(f"num_agents: {env.num_agents}")

        if len(env.agents) != len(actor_names) and len(actor_names) > 1:
            raise ValueError(f"Number of actors ({len(actor_names)}) does not match environments requirement ({len(env.agents)} actors)")

        pz_player_names = (
            {agent_name: 0 for agent_name in env.agents}
            if len(actor_names) == 1 and len(env.agents) > len(actor_names)
            else {agent_name: count for (count, agent_name) in enumerate(env.agents)}
        )

        assert len(web_actor_idx) < 2
        human_player_name = env.agents[web_actor_idx[0]] if web_actor_idx else ""
        pz_player_name = next(agent_iter)
        actor_idx = pz_player_names[pz_player_name]
        actor_name = actor_names[actor_idx]

        # Render the pixel for UI
        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in env.metadata["render_modes"]:
                log.warning(f"PettingZoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            rendered_frame = env.render()

        observation = observation_space.create(
            value=pz_observation,
            rendered_frame=rendered_frame,
            current_player=actor_name,
            game_player_name=human_player_name,
        )

        environment_session.start([("*", observation_space.serialize(observation))])
        async for event in environment_session.all_events():
            if not event.actions:
                continue

            # Action
            action_value = action_space.deserialize(event.actions[actor_idx].action).value

            # Observation
            env.step(action_value)
            pz_observation, pz_reward, termination, truncation, _ = env.last()

            # Actor names
            pz_player_name = next(agent_iter)
            actor_idx = pz_player_names[pz_player_name]
            actor_name = actor_names[actor_idx]

            observation = observation_space.create(
                value=pz_observation,
                rendered_frame=rendered_frame,
                current_player=actor_name,
                game_player_name=human_player_name,
            )
            rendered_frame = None
            if session_cfg.render:
                rendered_frame = env.render()

            observations = [("*", observation_space.serialize(observation))]
            # TODO: need to revise the actor name received the reward
            environment_session.add_reward(value=pz_reward, confidence=1.0, to=[actor_name])
            if termination or truncation:
                # The trial ended
                # log.info("Environement done")
                environment_session.end(observations)
            elif event.type != cogment.EventType.ACTIVE:
                # The trial termination has been requested
                environment_session.end(observations)
            else:
                # The trial is active
                environment_session.produce_observations(observations)
        env.close()
