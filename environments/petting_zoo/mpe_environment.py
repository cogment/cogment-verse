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
import os
from typing import Tuple

import cogment
import numpy as np
import supersuit as ss
from cogment.environment import EnvironmentSession
from data_pb2 import Player

from cogment_verse.constants import EVALUATOR_ACTOR_CLASS, PLAYER_ACTOR_CLASS, WEB_ACTOR_NAME, ActorClass, ActorSpecType
from cogment_verse.specs import ActorSpecs, EnvironmentSpecs
from cogment_verse.specs.ndarray_serialization import SerializationFormat, deserialize_ndarray
from cogment_verse.utils import import_class
from environments.petting_zoo.utils import PettingZooEnvType

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


WEB_COMPONENTS = {
    "pettingzoo.atari.pong_v3": "AtariPong.js",
    "pettingzoo.classic.connect_four_v3": "ConnectFour.js",
}



class MpeSpecType(Enum):
    """ Used to associate different environment specs to different actors.
    """
    DEFAULT = "player"
    GOOD = "mpe_good"
    ADVERSARY = "mpe_adversary"

    @classmethod
    def from_config(cls, spec_type_str: str):
        try:
            return cls(spec_type_str)
        except ValueError:
            raise ValueError(f"Actor specs type ({spec_type_str}) is not a supported type: [{', '.join([spec_type.value for spec_type in MpeSpecType])}]")


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


        assert len(env.possible_agents) >= 1
        self.good_specs = ActorSpecs.create(
            observation_space=env.observation_space(env.possible_agents[0]),
            action_space=env.action_space(env.possible_agents[0]),
            web_components_file=web_components_file,
            spec_type=MpeSpecType.GOOD,
        )

        self.adversary_specs = ActorSpecs.create(
            observation_space=env.observation_space(env.possible_agents[0]),
            action_space=env.action_space(env.possible_agents[0]),
            web_components_file=web_components_file,
            spec_type=MpeSpecType.ADVERSARY,
        )

        self.env_specs = EnvironmentSpecs.create_heterogeneous(
            num_players=sum([self.env_cfg.num_good, self.env_cfg.num_adversaries]),
            turn_based=False,
            actor_specs=[self.good_specs, self.adversary_specs]
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

    async def impl(self, environment_session: EnvironmentSession):
        actors = environment_session.get_active_actors()
        print(f"env service actors: {[(actor_idx, actor.actor_name, actor.actor_class_name) for actor_idx, actor in enumerate(actors)]}")

        player_actors = [
            (actor_idx, actor.actor_name, actor.actor_class_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        web_actor_idx = [actor_idx for actor_idx, actor_name, _ in player_actors if actor_name == WEB_ACTOR_NAME]
        print(f"web_actor_idx: {web_actor_idx}")
        session_cfg = environment_session.config

        # Initialize environment
        env = self.env_class.env(
            num_good=self.env_cfg.num_good,
            num_adversaries=self.env_cfg.num_adversaries,
            num_obstacles=self.env_cfg.num_obstacles,
            max_cycles=self.env_cfg.max_cycles,
            continuous_actions=self.env_cfg.continuous_actions,
            render_mode="rgb_array" if session_cfg.render else None,
        )
        good_observation_space = self.env_specs[MpeSpecType.GOOD.value].get_observation_space(session_cfg.render_width)
        good_action_space = self.env_specs[MpeSpecType.GOOD.value].get_action_space()

        adversary_observation_space = self.env_specs[MpeSpecType.ADVERSARY.value].get_observation_space(session_cfg.render_width)
        adversary_action_space = self.env_specs[MpeSpecType.ADVERSARY.value].get_action_space()

        # Reset environment
        env.reset(seed=session_cfg.seed)
        agent_iter = iter(env.agent_iter())
        pz_observation, _, _, _, _ = env.last()

        print(f"num_agents: {env.num_agents}")
        print(f"pz_observation: {pz_observation}")

        if len(env.agents) != len(player_actors) and len(player_actors) > 1:
            raise ValueError(f"Number of actors ({len(player_actors)}) does not match environments requirement ({len(env.agents)} actors)")

        pz_player_names = (
            {agent_name: 0 for agent_name in env.agents}
            if len(player_actors) == 1 and len(env.agents) > len(player_actors)
            else {agent_name: count for (count, agent_name) in enumerate(env.agents)}
        )

        assert len(web_actor_idx) < 2
        human_player_name = env.agents[web_actor_idx[0]] if web_actor_idx else ""
        pz_player_name = next(agent_iter)
        actor_idx = pz_player_names[pz_player_name]
        print(f"actor_idx: {actor_idx}")
        actor_name = player_actors[actor_idx][1]
        # TODO: adapt spec type
        human_player = Player(index=web_actor_idx[0], name=human_player_name, spec_type=ActorSpecType.DEFAULT.value) if web_actor_idx else None
        current_player = Player(index=actor_idx, name=actor_name, spec_type=ActorSpecType.DEFAULT.value)

        # Render the pixel for UI
        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in env.metadata["render_modes"]:
                log.warning(f"PettingZoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            rendered_frame = env.render()


        initial_observations = []
        for actor_idx, actor in enumerate(actors):
            if actor.actor_class_name == MpeSpecType.GOOD.value:
                observation = good_observation_space.create(
                    value=pz_observation,
                    rendered_frame=None,
                    current_player=current_player,
                    game_player_name=human_player,
                )
                initial_observations.append((actor.actor_name, good_observation_space.serialize(observation)))

            elif actor.actor_class_name == MpeSpecType.ADVERSARY.value:
                observation = adversary_observation_space.create(
                    value=pz_observation,
                    rendered_frame=None,
                    current_player=current_player,
                    game_player_name=human_player,
                )
                initial_observations.append((actor.actor_name, adversary_observation_space.serialize(observation)))

            elif actor.actor_class_name == ActorClass.OBSERVER.value:
                observation = good_observation_space.create(
                    value=pz_observation,
                    rendered_frame=rendered_frame,
                    current_player=current_player,
                    game_player_name=human_player,
                )
                initial_observations.append((actor.actor_name, good_observation_space.serialize(observation)))

            else:
                raise ValueError(f"Actor class {actor.actor_class_name} is not supported by the MpeEnvironment.")


        environment_session.start(initial_observations)
        async for event in environment_session.all_events():
            if not event.actions:
                continue

            # Action

            actor_spec_type = MpeSpecType.from_config()
            action_value = good_action_space.deserialize(event.actions[actor_idx].action).value

            # Observation
            env.step(action_value)
            pz_observation, pz_reward, termination, truncation, _ = env.last()

            # Actor names
            pz_player_name = next(agent_iter)
            actor_idx = pz_player_names[pz_player_name]
            actor_name = player_actors[actor_idx][1]
            current_player = Player(index=actor_idx, name=actor_name, spec_type=ActorSpecType.DEFAULT.value)

            observation = good_observation_space.create(
                value=pz_observation,
                rendered_frame=rendered_frame,
                current_player=current_player,
                game_player_name=human_player,
            )
            rendered_frame = None
            if session_cfg.render:
                rendered_frame = env.render()

            observations = [("*", good_observation_space.serialize(observation))]
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
