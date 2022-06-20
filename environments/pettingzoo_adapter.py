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

log = logging.getLogger(__name__)


def action_mask_from_pz_action_mask(pz_action_mask):
    return SpaceMask(
        properties=[
            SpaceMask.PropertyMask(
                discrete=[action_idx for action_idx, action in enumerate(pz_action_mask) if action == 1]
            )
        ]
    )


class PzEnvType(Enum):
    CLASSIC = "classic"


class Environment:
    def __init__(self, cfg):
        self.env_class_name = cfg.env_class_name
        env_type_str = self.env_class_name.split(".")[1]
        if env_type_str not in [pz_env_type.value for pz_env_type in PzEnvType]:
            raise RuntimeError(f"PettingZoo adapter does not support environments of type [{env_type_str}]")
        self.env_type = PzEnvType(env_type_str)
        self.env_class = import_class(self.env_class_name)

        pz_env = self.env_class.env()

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
            observation_space=space_from_gym_space(observation_space["observation"]),
            action_space=space_from_gym_space(action_space),
            turn_based=self.env_type in [PzEnvType.CLASSIC],
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
        assert len(player_actors) == self.env_specs.num_players  # pylint: disable=no-member

        # No support for teachers
        teacher_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == TEACHER_ACTOR_CLASS
        ]
        assert len(teacher_actors) == 0

        session_cfg = environment_session.config

        pz_env = self.env_class.env()

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

        current_player_pz_agent, current_player_actor_idx, current_player_actor_name = next_player()
        actor_last_tick_id = {}

        pz_observation, _pz_reward, _pz_done, _pz_info = pz_env.last()

        observation_value = observation_from_gym_observation(
            pz_env.observation_space(current_player_pz_agent)["observation"], pz_observation["observation"]
        )
        action_mask = action_mask_from_pz_action_mask(pz_observation["action_mask"])

        rendered_frame = None
        if session_cfg.render:
            if "rgb_array" not in pz_env.metadata["render_modes"]:
                log.warning(f"Petting Zoo environment [{self.env_class_name}] doesn't support rendering to pixels")
                return
            rendered_frame = encode_rendered_frame(pz_env.render(mode="rgb_array"), session_cfg.render_width)

        environment_session.start(
            [
                (
                    "*",
                    Observation(
                        value=observation_value,
                        rendered_frame=rendered_frame,
                        action_mask=action_mask,
                        current_player=current_player_actor_name,
                    ),
                )
            ]
        )

        async for event in environment_session.all_events():
            if event.actions:
                player_action_value = event.actions[current_player_actor_idx].action.value
                action_value = player_action_value
                # overridden_players = []
                # if has_teacher and event.actions[teacher_actor_idx].action.HasField("value"):
                #     teacher_action_value = event.actions[teacher_actor_idx].action.value
                #     action_value = teacher_action_value
                #     overridden_players = [player_actor_name]

                gym_action = gym_action_from_action(
                    self.env_specs.action_space, action_value  # pylint: disable=no-member
                )

                pz_env.step(gym_action)

                current_player_pz_agent, current_player_actor_idx, current_player_actor_name = next_player()
                pz_observation, _pz_reward, _pz_done, _pz_info = pz_env.last()

                observation_value = observation_from_gym_observation(
                    pz_env.observation_space(current_player_pz_agent)["observation"], pz_observation["observation"]
                )
                action_mask = action_mask_from_pz_action_mask(pz_observation["action_mask"])

                rendered_frame = None
                if session_cfg.render:
                    rendered_frame = encode_rendered_frame(pz_env.render(mode="rgb_array"), session_cfg.render_width)

                observations = [
                    (
                        "*",
                        Observation(
                            value=observation_value,
                            rendered_frame=rendered_frame,
                            action_mask=action_mask,
                            current_player=current_player_actor_name,
                        ),
                    )
                ]

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
                    assert rewarded_player_actor_name in actor_last_tick_id
                    environment_session.add_reward(
                        value=pz_reward,
                        confidence=1.0,
                        to=[rewarded_player_actor_name],
                        tick_id=actor_last_tick_id[rewarded_player_actor_name],
                    )

                actor_last_tick_id[current_player_actor_name] = environment_session.get_tick_id()

                done = all(pz_env.dones[pz_agent] for pz_agent in pz_env.agents)

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
