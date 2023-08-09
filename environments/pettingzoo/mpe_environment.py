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
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple

import cogment
import numpy as np
import supersuit as ss
from cogment.environment import EnvironmentSession
from data_pb2 import Player

from cogment_verse.constants import EVALUATOR_ACTOR_CLASS, PLAYER_ACTOR_CLASS, WEB_ACTOR_NAME, ActorClass, ActorSpecType
from cogment_verse.specs import ActorSpecs, EnvironmentSpecs
from cogment_verse.specs.ndarray_serialization import SerializationFormat, deserialize_ndarray
from cogment_verse.utils import import_class
from environments.pettingzoo.utils import MpeSpecType, PettingZooEnvType, get_strings_with_prefix

log = logging.getLogger(__name__)

WEB_COMPONENTS = {
    "pettingzoo.mpe.simple_tag_v3": "SimpleTag.js",
}

AGENT_PREFIX = "agent"
ADVERSARY_PREFIX = "adversary"

SPEC_TYPE_TO_PREFIX = {
    MpeSpecType.AGENT: AGENT_PREFIX,
    MpeSpecType.ADVERSARY: ADVERSARY_PREFIX,
}


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
        self._web_actor_spec = MpeSpecType.from_config(cfg.web_actor_spec) if "web_actor_spec" in cfg else MpeSpecType.AGENT
        self._player_classes = [PLAYER_ACTOR_CLASS] + MpeSpecType.values

        env = self.env_class.env(
            num_good=self.env_cfg.num_good,
            num_adversaries=self.env_cfg.num_adversaries,
            num_obstacles=self.env_cfg.num_obstacles,
            max_cycles=self.env_cfg.max_cycles,
            continuous_actions=self.env_cfg.continuous_actions
        )

        web_components_file = None
        matching_web_components = [
            web_component
            for (env_name_prefix, web_component) in WEB_COMPONENTS.items()
            if self.env_class_name.startswith(env_name_prefix)
        ]

        if len(matching_web_components) > 1:
            log.warning(
                f"While configuring petting zoo environment [{self.env_class_name}] found more that one matchingoofy_diffie_0_0g web components [{matching_web_components}], picking the first one."
            )
        if len(matching_web_components) > 0:
            web_components_file = matching_web_components[0]

        assert len(env.possible_agents) >= 1

        actor_specs = [
            ActorSpecs.create(
                observation_space=env.observation_spaces[get_strings_with_prefix(env.possible_agents, AGENT_PREFIX)[0]],
                action_space=env.action_spaces[get_strings_with_prefix(env.possible_agents, AGENT_PREFIX)[0]],
                web_components_file=web_components_file,
                spec_type=MpeSpecType.AGENT,
            ),
            ActorSpecs.create(
                observation_space=env.observation_spaces[get_strings_with_prefix(env.possible_agents, ADVERSARY_PREFIX)[0]],
                action_space=env.action_spaces[get_strings_with_prefix(env.possible_agents, ADVERSARY_PREFIX)[0]],
                web_components_file=web_components_file,
                spec_type=MpeSpecType.ADVERSARY,
            ),
            ActorSpecs.create(
                observation_space=env.observation_spaces[get_strings_with_prefix(env.possible_agents, SPEC_TYPE_TO_PREFIX[self._web_actor_spec])[0]],
                action_space=env.action_spaces[get_strings_with_prefix(env.possible_agents, SPEC_TYPE_TO_PREFIX[self._web_actor_spec])[0]],
                web_components_file=web_components_file,
                spec_type=MpeSpecType.DEFAULT,
            ),
        ]

        self.env_specs = EnvironmentSpecs.create_heterogeneous(
            num_players=sum([self.env_cfg.num_good, self.env_cfg.num_adversaries]),
            turn_based=False,
            actor_specs=actor_specs,
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
            if actor.actor_class_name in self._player_classes
        ]

        # Web actor index in cogment actors
        web_actor_idx = [actor_idx for actor_idx, actor_name, _ in player_actors if actor_name == WEB_ACTOR_NAME]
        assert len(web_actor_idx) < 2, "Multiple web actors are currently not supported for this environment."

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
        agent_observation_space = self.env_specs[MpeSpecType.AGENT.value].get_observation_space(session_cfg.render_width)
        adversary_observation_space = self.env_specs[MpeSpecType.ADVERSARY.value].get_observation_space(session_cfg.render_width)

        # Reset environment
        env.reset(seed=session_cfg.seed)
        agent_iter = iter(env.agent_iter())
        pz_observation, _, _, _, _ = env.last()

        if len(env.agents) != len(player_actors) and len(player_actors) > 1:
            raise ValueError(f"Number of actors ({len(player_actors)}) does not match environments requirement ({len(env.agents)} actors)")

        pz_name_to_idx = {agent_name: count for (count, agent_name) in enumerate(env.agents)}
        pz_idx_to_name = {v: k for k, v in pz_name_to_idx.items()}

        pz_to_actor_idx_mapping = {}
        for _actor_idx, _actor_name, _ in player_actors:
            if _actor_name == WEB_ACTOR_NAME:
                if self._web_actor_spec == MpeSpecType.ADVERSARY:
                    current_names = [pz_idx_to_name[idx] for idx, _ in pz_to_actor_idx_mapping.items()]
                    idx_count = len(get_strings_with_prefix(strings=current_names, prefix=ADVERSARY_PREFIX))
                    pz_web_actor_name = f"{ADVERSARY_PREFIX}_{idx_count}"
                else:
                    current_names = [pz_idx_to_name[idx] for idx, _ in pz_to_actor_idx_mapping.items()]
                    idx_count = len(get_strings_with_prefix(strings=current_names, prefix=AGENT_PREFIX))
                    pz_web_actor_name = f"{AGENT_PREFIX}_{idx_count}"

                pz_web_actor_idx = pz_name_to_idx[pz_web_actor_name]

            else:
                pz_web_actor_idx = pz_name_to_idx[_actor_name]

            pz_to_actor_idx_mapping[pz_web_actor_idx] = _actor_idx

        pz_player_name = next(agent_iter)

        pz_actor_idx = pz_name_to_idx[pz_player_name]
        cog_actor_idx = pz_to_actor_idx_mapping[pz_actor_idx]
        actor_name = player_actors[cog_actor_idx][1]
        actor_class_name = player_actors[cog_actor_idx][2]
        current_player = Player(index=cog_actor_idx, name=actor_name, spec_type=actor_class_name)

        # Render the pixel for UI
        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in env.metadata["render_modes"]:
                log.warning(f"PettingZoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            rendered_frame = env.render()

        initial_observations = []
        for _, actor in enumerate(actors):
            if actor.actor_class_name in [MpeSpecType.AGENT.value, MpeSpecType.DEFAULT.value]:
                observation = agent_observation_space.create(
                    value=pz_observation,
                    rendered_frame=None,
                    current_player=current_player,
                )
                initial_observations.append((actor.actor_name, agent_observation_space.serialize(observation)))

            elif actor.actor_class_name == MpeSpecType.ADVERSARY.value:
                observation = adversary_observation_space.create(
                    value=pz_observation,
                    rendered_frame=None,
                    current_player=current_player,
                )
                initial_observations.append((actor.actor_name, adversary_observation_space.serialize(observation)))

            elif actor.actor_class_name == ActorClass.OBSERVER.value:
                observation = agent_observation_space.create(
                    value=pz_observation,
                    rendered_frame=rendered_frame,
                    current_player=current_player,
                )
                initial_observations.append((actor.actor_name, agent_observation_space.serialize(observation)))

            else:
                raise ValueError(f"Actor class {actor.actor_class_name} is not supported by the MpeEnvironment.")


        environment_session.start(initial_observations)
        async for event in environment_session.all_events():
            if not event.actions:
                continue

            # Action
            action_space = self.env_specs[actor_class_name].get_action_space()

            # Only deserialize cogment action from current player according to pz env.
            action = action_space.deserialize(event.actions[cog_actor_idx].action)
            action_value = action.value

            # Observation (for next player)
            env.step(action_value)
            pz_observation, pz_reward, termination, truncation, _ = env.last()

            # Iterate to next player
            pz_player_name = next(agent_iter)                       # Next PettingZoo actor name
            pz_actor_idx = pz_name_to_idx[pz_player_name]           # Next PettingZoo actor index
            cog_actor_idx = pz_to_actor_idx_mapping[pz_actor_idx]   # Next Cogment actor index
            actor_name = player_actors[cog_actor_idx][1]            # Next Cogment actor name
            actor_class_name = player_actors[cog_actor_idx][2]      # Next Cogment actor class
            current_player = Player(index=cog_actor_idx, name=actor_name, spec_type=actor_class_name)

            observations = []
            for _, actor in enumerate(actors):
                if actor.actor_class_name in [MpeSpecType.AGENT.value, MpeSpecType.DEFAULT.value]:
                    observation = agent_observation_space.create(
                        value=pz_observation,
                        rendered_frame=rendered_frame,
                        current_player=current_player,
                        # human_player=human_player,
                    )
                    observations.append((actor.actor_name, agent_observation_space.serialize(observation)))

                elif actor.actor_class_name == MpeSpecType.ADVERSARY.value:
                    observation = adversary_observation_space.create(
                        value=pz_observation,
                        rendered_frame=None,
                        current_player=current_player,
                        # human_player=human_player,
                    )
                    observations.append((actor.actor_name, adversary_observation_space.serialize(observation)))

                elif actor.actor_class_name == ActorClass.OBSERVER.value:
                    observation = agent_observation_space.create(
                        value=pz_observation,
                        rendered_frame=rendered_frame,
                        current_player=current_player,
                        # human_player=human_player,
                    )
                    observations.append((actor.actor_name, agent_observation_space.serialize(observation)))

                else:
                    raise ValueError(f"Actor class {actor.actor_class_name} is not supported by the MpeEnvironment.")

            rendered_frame = None
            if session_cfg.render:
                rendered_frame = env.render()

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
