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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import cogment
import gymnasium as gym
import numpy as np
import supersuit as ss
from cogment.environment import EnvironmentSession
from data_pb2 import Player

from cogment_verse.constants import EVALUATOR_ACTOR_CLASS, PLAYER_ACTOR_CLASS, WEB_ACTOR_NAME, ActorSpecType
from cogment_verse.specs import ActorSpecs
from cogment_verse.specs.ndarray_serialization import SerializationFormat, deserialize_ndarray
from cogment_verse.utils import import_class

log = logging.getLogger(__name__)


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


def atari_env_wrapper(env: gym.Env) -> gym.Env:
    """Wrapper for atari env"""
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)

    return env


class PettingZooEnvType(Enum):
    ATARI = "atari"
    CLASSIC = "classic"
    MPE = "mpe"


WEB_COMPONENTS = {
    "pettingzoo.atari.pong_v3": "AtariPong.js",
    "pettingzoo.classic.connect_four_v3": "ConnectFour.js",
}


class Environment(ABC):
    def __init__(self, cfg):
        self.env_class_name = cfg.env_class_name
        self.env_type_str = self.env_class_name.split(".")[1]
        if self.env_type_str not in [env_type.value for env_type in PettingZooEnvType]:
            raise RuntimeError(f"PettingZoo adapter does not support environments of type [{self.env_type_str}]")

        self.env_type = PettingZooEnvType(self.env_type_str)
        self.env_class = import_class(self.env_class_name)
        env = self.env_class.env()
        if self.env_type == PettingZooEnvType.ATARI:
            env = atari_env_wrapper(env)
            serialization_format = SerializationFormat.NPY
        elif self.env_type in [PettingZooEnvType.CLASSIC, PettingZooEnvType.MPE]:
            serialization_format = SerializationFormat.STRUCTURED
        else:
            raise ValueError(f"PettingZoo environment type [{self.env_type_str}] does not exist")

        num_players = 0
        observation_space = None
        action_space = None
        for player in env.possible_agents:
            num_players += 1
            if observation_space is None:
                observation_space = env.observation_space(player)
                action_space = env.action_space(player)
            else:
                if observation_space != env.observation_space(player) or action_space != env.action_space(player):
                    raise RuntimeError(
                        "PettingZoo environment with heterogeneous action/observation spaces are not supported yet"
                    )

        web_components_file = None
        matching_web_components = [
            web_component
            for (env_name_prefix, web_component) in WEB_COMPONENTS.items()
            if self.env_class_name.startswith(env_name_prefix)
        ]
        if len(matching_web_components) > 1:
            log.warning(
                f"While configuring petting zoo environment [{self.env_class_name}] found more that one matching web components [{matching_web_components}], picking the first one."
            )
        if len(matching_web_components) > 0:
            web_components_file = matching_web_components[0]

        assert num_players >= 1
        self.env_specs = ActorSpecs.create_homogeneous(
            num_players=num_players,
            observation_space=observation_space,
            action_space=action_space,
            turn_based=self.env_type in [PettingZooEnvType.CLASSIC],
            serialization_format=serialization_format,
            web_components_file=web_components_file,
        )

    def get_web_components_dir(self):
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), "web", "dist")

    def get_implementation_name(self):
        return self.env_class_name

    def get_environment_specs(self):
        return self.env_specs

    @abstractmethod
    async def impl(self, environment_session: EnvironmentSession):
        raise NotImplementedError


class ClassicEnvironment(Environment):
    """Classic PettingZoo e.g., connect four, Hanabi etc."""

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        assert len(player_actors) == self.env_specs.num_players  # pylint: disable=no-member

        session_cfg = environment_session.config

        env = self.env_class.env(render_mode="rgb_array")
        observation_space = self.env_specs[ActorSpecType.DEFAULT].get_observation_space(session_cfg.render_width)
        action_space = self.env_specs[ActorSpecType.DEFAULT].get_action_space()

        env.reset(seed=session_cfg.seed)

        agent_iter = iter(env.agent_iter())

        def next_player():
            nonlocal agent_iter
            current_player_agent = next(agent_iter)
            current_player_actor_idx, current_player_actor_name = next(
                (player_actor_idx, player_actor_name)
                for (player_pz_agent, (player_actor_idx, player_actor_name)) in zip(env.agents, player_actors)
                if player_pz_agent == current_player_agent
            )
            return Player(index=current_player_actor_idx, name=current_player_actor_name, spec_type=ActorSpecType.DEFAULT.value)

        current_player = next_player()

        pz_observation, _pz_reward, termination, truncation, _ = env.last()

        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in env.metadata["render_modes"]:
                log.warning(f"PettingZoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            rendered_frame = env.render()

        observation = observation_space.create(
            value=pz_observation["observation"],  # TODO Should only be sent to the current player
            action_mask=pz_observation["action_mask"],  # TODO Should only be sent to the current player
            rendered_frame=rendered_frame,  # TODO Should only be sent to observers
            current_player=current_player,
        )

        environment_session.start([("*", observation_space.serialize(observation))])

        async for event in environment_session.all_events():
            if event.actions:
                action = action_space.deserialize(
                    event.actions[current_player.index].action,
                )

                env.step(action.value)

                current_player = next_player()
                pz_observation, _reward, termination, truncation, _ = env.last()

                observation = observation_space.create(
                    value=pz_observation["observation"],  # TODO Should only be sent to the current player
                    action_mask=pz_observation["action_mask"],  # TODO Should only be sent to the current player
                    rendered_frame=env.render()
                    if session_cfg.render
                    else None,  # TODO Should only be sent to observers
                    current_player=current_player,
                )

                observations = [("*", observation_space.serialize(observation))]

                for rewarded_player_pz_agent, reward in env.rewards.items():
                    if reward == 0:
                        continue
                    rewarded_player_actor_name = next(
                        player_actor_name
                        for (player_pz_agent, (player_actor_idx, player_actor_name)) in zip(
                            env.agents, player_actors
                        )
                        if player_pz_agent == rewarded_player_pz_agent
                    )
                    environment_session.add_reward(
                        value=reward,
                        confidence=1.0,
                        to=[rewarded_player_actor_name],
                    )

                if termination or truncation:
                    # The trial ended
                    environment_session.end(observations)
                elif event.type != cogment.EventType.ACTIVE:
                    # The trial termination has been requested
                    environment_session.end(observations)
                else:
                    # The trial is active
                    environment_session.produce_observations(observations)

        env.close()


class AtariEnvironment(Environment):
    async def impl(self, environment_session: EnvironmentSession):
        actors = environment_session.get_active_actors()
        actor_names = [actor.actor_name for actor in actors if actor.actor_class_name == PLAYER_ACTOR_CLASS]
        web_actor_idx = [count for (count, actor_name) in enumerate(actor_names) if actor_name == WEB_ACTOR_NAME]
        session_cfg = environment_session.config

        # Initialize environment
        env = self.env_class.env(render_mode="rgb_array") if session_cfg.render else self.env_class.env()
        env = atari_env_wrapper(env) if self.env_type_str == "atari" else env
        observation_space = self.env_specs[ActorSpecType.DEFAULT].get_observation_space(session_cfg.render_width)
        action_space = self.env_specs[ActorSpecType.DEFAULT].get_action_space()

        # Reset environment
        env.reset(seed=session_cfg.seed)
        agent_iter = iter(env.agent_iter())
        pz_observation, _, _, _, _ = env.last()

        if len(env.agents) != len(actor_names) and len(actor_names) > 1:
            raise ValueError(f"Number of actors does not match environments requirement ({len(env.agents)} actors)")

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

        human_player = Player(index=web_actor_idx[0], name=human_player_name, spec_type=ActorSpecType.DEFAULT.value)
        current_player = Player(index=actor_idx, name=actor_name, spec_type=ActorSpecType.DEFAULT.value)

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
            current_player=current_player,
            human_player=human_player,
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
            current_player = Player(index=actor_idx, name=actor_name, spec_type=ActorSpecType.DEFAULT.value)

            observation = observation_space.create(
                value=pz_observation,
                rendered_frame=rendered_frame,
                current_player=current_player,
                human_player=human_player,
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


class HumanFeedbackAtariEnvironment(Environment):
    async def impl(self, environment_session: EnvironmentSession):
        session_cfg = environment_session.config
        actors = environment_session.get_active_actors()
        valid_actor_class_names = {PLAYER_ACTOR_CLASS, EVALUATOR_ACTOR_CLASS}
        actor_names = [actor.actor_name for actor in actors if actor.actor_class_name in valid_actor_class_names]

        # Initialize environment
        pz_env = self.env_class.env(render_mode="rgb_array") if session_cfg.render else self.env_class.env()
        pz_env = atari_env_wrapper(pz_env) if self.env_type_str == "atari" else pz_env
        observation_space = self.env_specs[ActorSpecType.DEFAULT].get_observation_space(session_cfg.render_width)
        action_space = self.env_specs[ActorSpecType.DEFAULT].get_action_space()

        # Reset environment
        pz_env.reset(seed=session_cfg.seed)
        pz_agent_iterator = iter(pz_env.agent_iter())
        pz_observation, _, _, _, _ = pz_env.last()

        if (len(actor_names) != len(pz_env.agents) + 1) and len(actor_names) > 1:
            raise ValueError(f"Number of actors does not match environments requirement ({len(pz_env.agents)} actors)")
        pz_player_names = {agent_name: count for count, agent_name in enumerate(pz_env.agents)}
        pz_player_name = next(pz_agent_iterator)
        actor_idx = pz_player_names[pz_player_name]
        actor_name = actor_names[actor_idx]
        current_player = Player(index=actor_idx, name=actor_name, spec_type=ActorSpecType.DEFAULT.value)

        # Render the pixel for UI
        rendered_frame = None
        if session_cfg.render and "rgb_array" not in pz_env.metadata["render_modes"]:
            log.warning(f"PettingZoo environment [{self.env_class_name}] doesn't support rendering to pixels")
            return
        rendered_frame = pz_env.render()

        # Initial observation
        observation = observation_space.create(
            value=pz_observation,
            rendered_frame=rendered_frame,
            current_player=current_player,
            human_player=current_player,
            action_value=0,
        )
        environment_session.start([("*", observation_space.serialize(observation))])

        rewarded_actor_name = actor_name
        async for event in environment_session.all_events():
            if not event.actions:
                continue

            action_value = event.actions[actor_idx].action
            if actors[actor_idx].actor_class_name == PLAYER_ACTOR_CLASS:
                # Observation
                gym_action = action_space.deserialize(action_value)
                pz_env.step(gym_action.value)
                pz_observation, pz_reward, termination, truncation, _ = pz_env.last()

                # Actor names for evaluator
                rewarded_actor_name = actor_name
                reward_actor = Player(index=actor_idx, name=rewarded_actor_name, spec_type=ActorSpecType.DEFAULT.value)

                actor_idx = -1
                actor_name = actor_names[actor_idx]
                current_player = Player(index=actor_idx, name=actor_name, spec_type=ActorSpecType.DEFAULT.value)

                # Pixel frame display on UI
                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = pz_env.render()
            elif actors[actor_idx].actor_class_name == EVALUATOR_ACTOR_CLASS:
                # TODO: To be modified when sending-UI-reward is added
                pz_reward = deserialize_ndarray(action_value.value)[0]
                pz_player_name = next(pz_agent_iterator)
                actor_idx = pz_player_names[pz_player_name]
                actor_name = actor_names[actor_idx]
                current_player = Player(index=actor_idx, name=actor_name, spec_type=ActorSpecType.DEFAULT.value)
            else:
                # Raise an error if the actor class is invalid
                raise ValueError("Actor class is invalid")

            observation = observation_space.create(
                value=pz_observation,
                rendered_frame=rendered_frame,
                current_player=current_player,
                human_player=reward_actor,
                action_value=gym_action.value,
            )

            observations = [("*", observation_space.serialize(observation))]
            environment_session.add_reward(value=pz_reward, confidence=1.0, to=[rewarded_actor_name])

            if termination or truncation:
                # The trial ended
                environment_session.end(observations)
            elif event.type != cogment.EventType.ACTIVE:
                # The trial termination has been requested
                environment_session.end(observations)
            else:
                # The trial is active
                environment_session.produce_observations(observations)

        pz_env.close()
