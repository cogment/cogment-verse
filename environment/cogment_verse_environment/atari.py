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
from cogment_verse_environment.base import GymObservation
from cogment_verse_environment.env_spec import EnvSpec
from cogment_verse_environment.gym_env import GymEnv
from gym.envs import register

# Atari-py includes a free Tetris rom for testing without needing to download other ROMs
register(
    id="TetrisALENoFrameskip-v0",
    entry_point="gym.envs.atari:AtariEnv",
    kwargs={"game": "tetris", "obs_type": "image"},
    max_episode_steps=10000,
    nondeterministic=False,
)


def _grayscale(image):
    return np.mean(image, axis=2)


class AtariEnv(GymEnv):
    """
    Class for loading Atari environments.
    """

    def __init__(
        self,
        *,
        env_name,
        frame_skip=4,
        screen_size=84,
        sticky_actions=True,
        flatten=True,
        framestack=4,
        **_kwargs,
    ):
        """
        Args:
            env_name (str): Name of the environment
            sticky_actions (boolean): Whether to use sticky_actions as per Machado et al.
            frame_skip (int): Number of times the agent takes the same action in the environment
            screen_size (int): Size of the resized frames from the environment
        """
        env_version = "v0" if sticky_actions else "v4"
        full_env_name = f"{env_name}NoFrameskip-{env_version}"

        self._framestack = framestack

        if frame_skip <= 0:
            raise ValueError(f"Frame skip should be strictly positive, got {frame_skip}")
        if screen_size <= 0:
            raise ValueError(f"Target screen size should be strictly positive, got {screen_size}")

        self.frame_skip = frame_skip
        self.screen_size = screen_size

        self._flatten = flatten
        self._last_obs = []  # to be used for framestacking

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
        # Used for storing and pooling over two consecutive observations to reduce flicker
        self.screen_buffer = [_grayscale(self._env.reset())] * 2
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
        flicker_frames = 2

        for _ in range(self.frame_skip):
            observation, reward, done, info = self._env.step(action)
            observation = _grayscale(observation)
            accumulated_reward += reward

            self.screen_buffer = [observation] + self.screen_buffer
            self.screen_buffer = self.screen_buffer[:flicker_frames]

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

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        Returns:
          transformed_screen (numpy array): pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[1])

        transformed_image = cv2.resize(
            self.screen_buffer[1],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        if self._flatten:
            int_image = int_image.reshape(-1)
        return np.expand_dims(int_image, axis=0)
