# Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import cv2
import numpy as np

# registers procgen environments
import procgen  # pylint: disable=unused-import
from cogment_verse_environment.base import GymObservation
from cogment_verse_environment.env_spec import EnvSpec
from cogment_verse_environment.gym_env import GymEnv

ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


def _grayscale(image):
    return np.mean(image, axis=2)


class ProcGenEnv(GymEnv):
    """
    Class for loading procgen environments.
    """

    def __init__(
        self,
        *,
        env_name,
        frame_skip=1,
        screen_size=64,
        flatten=True,
        framestack=4,
        **_kwargs,
    ):
        full_env_name = f"procgen-{env_name}-v0"

        self._framestack = framestack

        if frame_skip <= 0:
            raise ValueError(f"Frame skip should be strictly positive, got {frame_skip}")
        if screen_size <= 0:
            raise ValueError(f"Target screen size should be strictly positive, got {screen_size}")

        self.frame_skip = frame_skip
        self.screen_size = screen_size

        self._flatten = flatten
        self._last_obs = []  # to be used for framestacking
        self._last_pixels = None

        super().__init__(env_name=full_env_name, num_players=1, framestack=framestack)

    def create_env_spec(self, env_name, **_kwargs):
        act_spaces = [self._env.action_space]
        return EnvSpec(
            env_name=env_name,
            obs_dim=[(self._framestack, self.screen_size, self.screen_size)],
            act_dim=[space.n for space in act_spaces],
            act_shape=[()],
        )

    def _prepare_obs(self):
        obs = np.concatenate(self._last_obs)
        if self._flatten:
            obs = obs.reshape(-1)
        return obs

    def _state(self):
        obs = self._pool_and_resize()
        return obs, self._turn

    def reset(self):
        self._last_pixels = self._env.reset()
        observation, current_player = self._state()

        if self._framestack > 1:
            self._last_obs = [observation] * self._framestack
        else:
            self._last_obs = [observation]

        return GymObservation(
            observation=self._prepare_obs(),
            current_player=current_player,
            legal_moves_as_int=[],
            rewards=[0.0],
            done=False,
            info={},
        )

    def step(self, action=None):
        """
        Remarks:
            * Execute self.frame_skips steps taking the action in the the environment.
            * This may execute fewer than self.frame_skip steps in the environment, if the done state is reached.
            * Furthermore, in this case the returned observation should be ignored.
        """
        assert action is not None

        accumulated_reward = 0.0
        done = False
        info = {}

        for _ in range(self.frame_skip):
            observation, reward, done, info = self._env.step(action)
            self._last_pixels = observation
            observation = _grayscale(observation)
            accumulated_reward += reward

            if done:
                break

        observation, current_player = self._state()

        if self._framestack > 1:
            self._last_obs = [observation] + self._last_obs[:-1]
        else:
            self._last_obs = [observation]

        return GymObservation(
            observation=self._prepare_obs(),
            current_player=current_player,
            legal_moves_as_int=[],
            rewards=[accumulated_reward],
            done=done,
            info=info,
        )

    def _pool_and_resize(self, dtype=np.uint8):
        """Transforms two frames into a Nature DQN observation.

        Returns:
          transformed_screen (numpy array): pooled, resized screen.
        """
        image = _grayscale(self._last_pixels)
        if image.shape != (self.screen_size, self.screen_size):
            image = cv2.resize(
                image,
                (self.screen_size, self.screen_size),
                interpolation=cv2.INTER_AREA,
            )
        image = np.asarray(image, dtype=dtype)
        if self._flatten:
            image = image.reshape(-1)
        return np.expand_dims(image, axis=0)

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        return self._last_pixels

    def seed(self, seed=None):
        # self._env.seed(seed)
        # todo
        pass
