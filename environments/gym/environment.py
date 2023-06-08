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

from cogment_verse.constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS
from cogment_verse.specs import EnvironmentSpecs

log = logging.getLogger(__name__)

# configure pygame to use a dummy video server to be able to render headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

WEB_COMPONENTS = {
    "ALE/": "Atari.js",
    "CartPole": "CartPole.js",
    "LunarLander": "LunarLander.js",
    "LunarLanderContinuous": "LunarLanderContinuous.js",
    "MountainCar": "MountainCar.js",
}


class Environment:
    def __init__(self, cfg):
        self.gym_env_name = cfg.env_name

        gym_env = gym.make(self.gym_env_name, new_step_api=True)

        web_components_file = None
        matching_web_components = [
            web_component
            for (env_name_prefix, web_component) in WEB_COMPONENTS.items()
            if self.gym_env_name.startswith(env_name_prefix)
        ]
        if len(matching_web_components) > 1:
            log.warning(
                f"While configuring gym environment [{self.gym_env_name}] found more that one matching web components [{matching_web_components}], picking the first one."
            )
        if len(matching_web_components) > 0:
            web_components_file = matching_web_components[0]

        self.env_specs = EnvironmentSpecs.create_homogeneous(
            num_players=1,
            turn_based=False,
            observation_space=gym_env.observation_space,
            action_space=gym_env.action_space,
            web_components_file=web_components_file,
        )

        if self.gym_env_name.startswith("ALE"):
            # Atari environment
            self._make_gym_env = lambda render: gym.make(
                self.gym_env_name,
                render_mode="rgb_array" if render else None,
                new_step_api=True,
                full_action_space=True,
            )
            self._render_gym_env = lambda gym_env: gym_env.render(mode="rgb_array")
        else:
            self._make_gym_env = lambda render: gym.make(
                self.gym_env_name,
                render_mode="single_rgb_array" if render else None,
                new_step_api=True,
            )
            self._render_gym_env = lambda gym_env: gym_env.render()

    def get_web_components_dir(self):
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), "web", "dist")

    def get_implementation_name(self):
        return self.gym_env_name

    def get_environment_specs(self):
        return self.env_specs

    async def impl(self, environment_session):
        actors = environment_session.get_active_actors()
        player_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == PLAYER_ACTOR_CLASS
        ]
        assert len(player_actors) == 1
        [(player_actor_idx, player_actor_name)] = player_actors

        teacher_actors = [
            (actor_idx, actor.actor_name)
            for (actor_idx, actor) in enumerate(actors)
            if actor.actor_class_name == TEACHER_ACTOR_CLASS
        ]
        assert len(teacher_actors) <= 1
        has_teacher = len(teacher_actors) == 1

        if has_teacher:
            [(teacher_actor_idx, _teacher_actor_name)] = teacher_actors

        session_cfg = environment_session.config

        gym_env = self._make_gym_env(session_cfg.render)
        observation_space = self.env_specs.get_observation_space(session_cfg.render_width)
        action_space = self.env_specs.get_action_space()

        gym_observation, _info = gym_env.reset(seed=session_cfg.seed, return_info=True)

        observation = observation_space.create(
            value=gym_observation,
            rendered_frame=self._render_gym_env(gym_env) if session_cfg.render else None,
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

                # Clipped action and send to gym environment
                if isinstance(gym_env.action_space, gym.spaces.Box):
                    action_value = np.clip(action_value, gym_env.action_space.low, gym_env.action_space.high)

                gym_observation, reward, terminated, truncated, _info = gym_env.step(action_value)

                observation = observation_space.create(
                    value=gym_observation,
                    rendered_frame=self._render_gym_env(gym_env) if session_cfg.render else None,
                    overridden_players=overridden_players,
                )

                observations = [("*", observation_space.serialize(observation))]

                if reward is not None:
                    environment_session.add_reward(
                        value=reward,
                        confidence=1.0,
                        to=[player_actor_name],
                    )

                if terminated or truncated:
                    # The trial ended
                    environment_session.end(observations)
                elif event.type != cogment.EventType.ACTIVE:
                    # The trial termination has been requested
                    environment_session.end(observations)
                else:
                    # The trial is active
                    environment_session.produce_observations(observations)

        gym_env.close()
