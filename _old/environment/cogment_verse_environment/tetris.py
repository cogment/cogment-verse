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

import gym_tetris
import numpy as np
from cogment_verse_environment.atari import AtariEnv, GymEnv
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace


class TetrisEnv(AtariEnv):
    """
    Class for loading gym built-in environments.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        *,
        env_name,
        frame_skip=4,
        screen_size=84,
        _sticky_actions=True,
        flatten=True,
        framestack=4,
        **_kwargs,
    ):  # pylint: disable=super-init-not-called,non-parent-init-called
        """
        Args:
            env_name: Name of the environment (NOTE: make sure it is available at gym.envs.registry.all())
        """
        if frame_skip <= 0:
            raise ValueError(f"Frame skip should be strictly positive, got {frame_skip}")
        if screen_size <= 0:
            raise ValueError(f"Target screen size should be strictly positive, got {screen_size}")

        self.frame_skip = frame_skip
        self.screen_size = screen_size

        self._framestack = framestack
        self._flatten = flatten

        GymEnv.__init__(self, env_name=env_name, num_players=1, framestack=framestack)

    def create_env(self, env_name, **_kwargs):
        """Function used to create the environment. Subclasses can override this method
        if they are using a gym style environment that needs special logic.
        """
        env = gym_tetris.make(env_name)
        self._env = JoypadSpace(env, MOVEMENT)

    def _get_observation_screen(self, output):  # pylint: disable=no-self-use
        """Get the screen input of the current observation given empty numpy array in grayscale.

        Args:
          output (numpy array): screen buffer to hold the returned observation.

        Returns:
          observation (numpy array): the current observation in grayscale.
        """
        # self._env.ale.getScreenGrayscale(output)
        return output

    def step(self, action=None):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        return super().step(action)
