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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import cogment
import gymnasium as gymna
import numpy as np
import supersuit as ss
from cogment.environment import EnvironmentSession

from cogment_verse.constants import WEB_ACTOR_NAME, ActorClass
from cogment_verse.specs import EnvironmentSpecs
from cogment_verse.specs.ndarray_serialization import SerializationFormat, deserialize_ndarray
from cogment_verse.utils import import_class

log = logging.getLogger(__name__)


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


class Environment(ABC):
    def __init__(self, cfg):
        self.env_class_name = cfg.env_class_name
        self.env_type_str = self.env_class_name.split(".")[1]
        if self.env_type_str not in [pz_env_type.value for pz_env_type in PzEnvType]:
            raise RuntimeError(f"PettingZoo adapter does not support environments of type [{self.env_type_str}]")

        self.env_type = PzEnvType(self.env_type_str)
        self.env_class = import_class(self.env_class_name)
        pz_env = self.env_class.env()
        if self.env_type == PzEnvType.ATARI:
            pz_env = atari_env_wrapper(pz_env)
            serilization_format = SerializationFormat.NPY
        elif self.env_type == PzEnvType.CLASSIC:
            serilization_format = SerializationFormat.STRUCTURED
        else:
            raise ValueError("Petting zoo environment type does not exist")

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

        print(f"NUM PLAYERS: {num_players}")
        print(f"Turn-Based: {self.env_type in [PzEnvType.CLASSIC]}")

        assert num_players >= 1
        self.env_specs = EnvironmentSpecs.create_homogeneous(
            num_players=num_players,
            observation_space=observation_space,
            action_space=action_space,
            turn_based=self.env_type in [PzEnvType.CLASSIC],
            serilization_format=serilization_format,
        )

    def get_implementation_name(self):
        return self.env_class_name

    def get_environment_specs(self):
        return self.env_specs

    @abstractmethod
    async def impl(self, environment_session: EnvironmentSession):
        raise NotImplementedError


class ClassicEnvironment(Environment):
    """Classic petting zoo e.g., connect four, Hanabi etc."""

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == ActorClass.PLAYER.value
        ]
        assert len(player_actors) == self.env_specs.num_players  # pylint: disable=no-member

        # No support for teachers
        teacher_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == ActorClass.TEACHER.value
        ]
        assert len(teacher_actors) == 0

        session_cfg = environment_session.config

        pz_env = self.env_class.env(render_mode="rgb_array")
        observation_space = self.env_specs.get_observation_space(session_cfg.render_width)
        action_space = self.env_specs.get_action_space()

        pz_env.reset(seed=session_cfg.seed)

        pz_agent_iterator = iter(pz_env.agent_iter())

        def next_player():
            nonlocal pz_agent_iterator
            current_player_pz_agent = next(pz_agent_iterator)
            current_player_actor_idx, current_player_actor_name = next(
                (player_actor_idx, player_actor_name)
                for (player_pz_agent, (player_actor_idx, player_actor_name)) in zip(pz_env.agents, player_actors)
                if player_pz_agent == current_player_pz_agent
            )
            return (current_player_pz_agent, current_player_actor_idx, current_player_actor_name)

        _current_player_pz_agent, current_player_actor_idx, current_player_actor_name = next_player()

        pz_observation, _pz_reward, done, _, _ = pz_env.last()

        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in pz_env.metadata["render_modes"]:
                log.warning(f"Petting Zoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            rendered_frame = pz_env.render()

        observation = observation_space.create(
            value=pz_observation["observation"],  # TODO Should only be sent to the current player
            action_mask=pz_observation["action_mask"],  # TODO Should only be sent to the current player
            rendered_frame=rendered_frame,  # TODO Should only be sent to observers
            current_player=current_player_actor_name,
        )

        environment_session.start([("*", observation_space.serialize(observation))])

        async for event in environment_session.all_events():
            if event.actions:
                action = action_space.deserialize(
                    event.actions[current_player_actor_idx].action,
                )

                pz_env.step(action.value)

                _current_player_pz_agent, current_player_actor_idx, current_player_actor_name = next_player()
                pz_observation, _pz_reward, done, _, _ = pz_env.last()

                observation = observation_space.create(
                    value=pz_observation["observation"],  # TODO Should only be sent to the current player
                    action_mask=pz_observation["action_mask"],  # TODO Should only be sent to the current player
                    rendered_frame=pz_env.render()
                    if session_cfg.render
                    else None,  # TODO Should only be sent to observers
                    current_player=current_player_actor_name,
                )

                observations = [("*", observation_space.serialize(observation))]

                for (rewarded_player_pz_agent, pz_reward) in pz_env.rewards.items():
                    if pz_reward == 0:
                        continue
                    rewarded_player_actor_name = next(
                        player_actor_name
                        for (player_pz_agent, (player_actor_idx, player_actor_name)) in zip(
                            pz_env.agents, player_actors
                        )
                        if player_pz_agent == rewarded_player_pz_agent
                    )
                    environment_session.add_reward(
                        value=pz_reward,
                        confidence=1.0,
                        to=[rewarded_player_actor_name],
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


class AtariEnvironment(Environment):
    async def impl(self, environment_session: EnvironmentSession):
        actors = environment_session.get_active_actors()
        actor_names = [actor.actor_name for actor in actors if actor.actor_class_name == ActorClass.PLAYER.value]
        web_actor_idx = [count for (count, actor_name) in enumerate(actor_names) if actor_name == WEB_ACTOR_NAME]
        session_cfg = environment_session.config

        # Initialize environment
        pz_env = self.env_class.env(render_mode="rgb_array") if session_cfg.render else self.env_class.env()
        pz_env = atari_env_wrapper(pz_env) if self.env_type_str == "atari" else pz_env
        observation_space = self.env_specs.get_observation_space(session_cfg.render_width)
        action_space = self.env_specs.get_action_space()

        # Reset environment
        pz_env.reset(seed=session_cfg.seed)
        pz_agent_iterator = iter(pz_env.agent_iter())
        pz_observation, _, _, _, _ = pz_env.last()

        print(f"pz_observation shape: {pz_observation.size}")

        if len(pz_env.agents) != len(actor_names) and len(actor_names) > 1:
            raise ValueError(f"Number of actors does not match environments requirement ({len(pz_env.agents)} actors)")

        pz_player_names = (
            {agent_name: 0 for agent_name in pz_env.agents}
            if len(actor_names) == 1 and len(pz_env.agents) > len(actor_names)
            else {agent_name: count for (count, agent_name) in enumerate(pz_env.agents)}
        )

        assert len(web_actor_idx) < 2
        human_player_name = pz_env.agents[web_actor_idx[0]] if web_actor_idx else ""
        pz_player_name = next(pz_agent_iterator)
        actor_idx = pz_player_names[pz_player_name]
        actor_name = actor_names[actor_idx]

        # Render the pixel for UI
        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in pz_env.metadata["render_modes"]:
                log.warning(f"Petting Zoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            rendered_frame = pz_env.render()

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
            action_value = event.actions[actor_idx].action

            # Observation
            gym_action = action_space.deserialize(action_value)
            pz_env.step(gym_action.value)
            pz_observation, pz_reward, done, _, _ = pz_env.last()

            # Actor names
            pz_player_name = next(pz_agent_iterator)
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
                rendered_frame = pz_env.render()

            observations = [("*", observation_space.serialize(observation))]
            # TODO: need to revise the actor name received the reward
            environment_session.add_reward(value=pz_reward, confidence=1.0, to=[actor_name])
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


class HumanFeedbackAtariEnvironment(Environment):
    async def impl(self, environment_session: EnvironmentSession):
        session_cfg = environment_session.config
        actors = environment_session.get_active_actors()
        valid_actor_class_names = {ActorClass.PLAYER.value, ActorClass.EVALUATOR.value}
        actor_names = [actor.actor_name for actor in actors if actor.actor_class_name in valid_actor_class_names]

        # Initialize environment
        pz_env = self.env_class.env(render_mode="rgb_array") if session_cfg.render else self.env_class.env()
        pz_env = atari_env_wrapper(pz_env) if self.env_type_str == "atari" else pz_env
        observation_space = self.env_specs.get_observation_space(session_cfg.render_width)
        action_space = self.env_specs.get_action_space()

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

        # Render the pixel for UI
        rendered_frame = None
        if session_cfg.render and "rgb_array" not in pz_env.metadata["render_modes"]:
            log.warning(f"Petting Zoo environment [{self.env_class_name}] doesn't support rendering to pixels")
            return
        rendered_frame = pz_env.render()

        # Initial observation
        observation = observation_space.create(
            value=pz_observation,
            rendered_frame=rendered_frame,
            current_player=actor_name,
            game_player_name=actor_name,
            action_value=0,
        )
        environment_session.start([("*", observation_space.serialize(observation))])

        rewarded_actor_name = actor_name
        async for event in environment_session.all_events():
            if not event.actions:
                continue

            action_value = event.actions[actor_idx].action
            if actors[actor_idx].actor_class_name == ActorClass.PLAYER.value:
                # Observation
                gym_action = action_space.deserialize(action_value)
                pz_env.step(gym_action.value)
                pz_observation, pz_reward, done, _, _ = pz_env.last()

                # Actor names for evaluator
                rewarded_actor_name = actor_name
                actor_idx = -1
                actor_name = actor_names[actor_idx]

                # Pixel frame display on UI
                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = pz_env.render()
            elif actors[actor_idx].actor_class_name == ActorClass.EVALUATOR.value:
                # TODO: To be modified when sending-UI-reward is added
                pz_reward = deserialize_ndarray(action_value.value)[0]
                pz_player_name = next(pz_agent_iterator)
                actor_idx = pz_player_names[pz_player_name]
                actor_name = actor_names[actor_idx]
            else:
                # Raise an error if the actor class is invalid
                raise ValueError("Actor class is invalid")

            observation = observation_space.create(
                value=pz_observation,
                rendered_frame=rendered_frame,
                current_player=actor_name,
                game_player_name=rewarded_actor_name,
                action_value=gym_action.value,
            )

            observations = [("*", observation_space.serialize(observation))]
            environment_session.add_reward(value=pz_reward, confidence=1.0, to=[rewarded_actor_name])
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
