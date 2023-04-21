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

# pylint: disable=C0303
# pylint: disable=E0401

import os

import cogment
import isaacgymenvs
import numpy as np
import torch

from cogment_verse.constants import ActorClass
from cogment_verse.specs import EnvironmentSpecs

# configure pygame to use a dummy video server to be able to render headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"


class Environment:
    def __init__(self, cfg):
        print("cfg = ", cfg)
        self.gym_env_name = cfg.env_name
        print("self.gym_env_name = ", self.gym_env_name)

        self.gym_env = isaacgymenvs.make(
            seed=0,
            task=self.gym_env_name,
            num_envs=1,
            sim_device="cuda:0",
            rl_device="cuda:0",
        )
        self.env_specs = EnvironmentSpecs.create_homogeneous(
            num_players=1,
            turn_based=False,
            observation_space=self.gym_env.observation_space,
            action_space=self.gym_env.action_space,
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
            if actor.actor_class_name == ActorClass.PLAYER.value
        ]
        assert len(player_actors) == 1
        [(player_actor_idx, player_actor_name)] = player_actors

        teacher_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == ActorClass.TEACHER.value
        ]
        assert len(teacher_actors) <= 1
        has_teacher = len(teacher_actors) == 1
        if has_teacher:
            [(teacher_actor_idx, _teacher_actor_name)] = teacher_actors

        session_cfg = environment_session.config

        observation_space = self.env_specs.get_observation_space(session_cfg.render_width)
        action_space = self.env_specs.get_action_space()

        gym_observation = self.gym_env.reset()
        obs = np.asarray(gym_observation["obs"].cpu())

        observation = observation_space.create(
            value=gym_observation,
            rendered_frame=self.gym_env.render(mode="rgb_array") if session_cfg.render else None,
        )

        environment_session.start([("*", observation_space.serialize(observation))])

        async for event in environment_session.all_events():
            if event.actions:
                player_action = action_space.deserialize(
                    event.actions[player_actor_idx].action,
                )
                action = player_action
                overridden_players = []
                if has_teacher:
                    teacher_action = action_space.deserialize(
                        event.actions[teacher_actor_idx].action,
                    )
                    if teacher_action.value is not None:
                        action = teacher_action
                        overridden_players = [player_actor_name]

                action_value = action.value

                gym_action_tensor = torch.tensor(action_value, device="cuda:0").view(-1, 8)
                gym_observation, reward, done, _info = self.gym_env.step(gym_action_tensor)
                obs = np.asarray(gym_observation["obs"].cpu())

                observation = observation_space.create(
                    value=obs,
                    rendered_frame=self.gym_env.render(mode="rgb_array") if session_cfg.render else None,
                    overridden_players=overridden_players,
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

        # self.gym_env.close()
