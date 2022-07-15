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

# Workaround for minatar's unnecessary tkagg dependency
# (this needs to be done before the minatar import)
# pylint: disable=wrong-import-position
import matplotlib
import numpy as np
from cogment_verse_environment.base import BaseEnv, GymObservation
from cogment_verse_environment.env_spec import EnvSpec

matplotlib.use("Agg")
matplotlib_use = matplotlib.use
matplotlib.use = lambda *args, **kwargs: None
import matplotlib.pyplot
from minatar.environment import Environment

matplotlib.use = matplotlib_use


class MinAtarEnv(BaseEnv):
    """
    Class for loading Atari environments.
    """

    def __init__(
        self,
        *,
        env_name,
        sticky_action_prob=0.1,
        difficulty_ramping=True,
        random_seed=None,
        flatten=True,
        num_players=1,
        framestack=4,
        **_kwargs,
    ):
        """
        Args:
            env_name (str): Name of the environment
            sticky_actions (boolean): Whether to use sticky_actions as per Machado et al.

        Available environments:
            asterix
            breakout
            freeway
            seaquest
            space_invaders
        """
        assert num_players == 1
        self._env = Environment(
            env_name,
            sticky_action_prob=sticky_action_prob,
            difficulty_ramping=difficulty_ramping,
            random_seed=random_seed,
        )
        self._flatten_obs = flatten
        self._last_state = []
        super().__init__(env_spec=self.create_env_spec(env_name), num_players=1, framestack=framestack)

    def create_env_spec(self, env_name):
        obs_dim = tuple(self._env.state_shape())
        new_positions = [2, 0, 1]
        obs_dim = tuple(obs_dim[i] for i in new_positions)
        return EnvSpec(env_name=env_name, obs_dim=[obs_dim], act_dim=[6], act_shape=[()])

    def _state(self):
        state = np.transpose(self._env.state(), [2, 1, 0])
        if self._flatten_obs:
            state = state.reshape(-1)
        return state

    def seed(self, seed=None):
        # TODO make that work, in minatar the seed should be provided in the constructor (cf. https://github.com/kenjyoung/MinAtar/blob/master/minatar/environment.py#L18-L27)
        pass

    def reset(self):
        self._env.reset()
        obs = self._state()

        if self._framestack > 1:
            self._last_obs = [obs] * self._framestack
        else:
            self._last_obs = obs

        return GymObservation(
            observation=np.concatenate(self._last_obs),
            current_player=0,
            legal_moves_as_int=[],
            rewards=[0.0],
            done=False,
            info={},
        )

    def step(self, action=None):
        """
        Remarks:
            * Execute self.frame_skips steps taking the action in the the environment.
            * This may execute fewer than self.frame_skip steps in the environment,
            if the done state is reached.
            * Furthermore, in this case the returned observation should be ignored.
        """
        assert action is not None
        reward, done = self._env.act(action)

        obs = self._state()

        if self._framestack > 1:
            self._last_obs = [obs] + self._last_obs[:-1]
        else:
            self._last_obs = obs

        return GymObservation(
            observation=np.concatenate(self._last_obs),
            current_player=0,
            legal_moves_as_int=[],
            rewards=[float(reward)],
            done=done,
            info={},
        )

    def close(self):
        pass
        # self._env.close_display()

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        state = self._env.state()
        assert state.shape == (10, 10, 4)
        canvas = np.zeros(shape=(10, 10, 3), dtype=np.uint8)

        colors = [[127, 127, 127], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

        colors = np.array(colors).reshape(4, 1, 1, 3)

        for color_idx in range(4):
            canvas = canvas + state[:, :, color_idx].reshape(10, 10, 1) * colors[color_idx]

        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

        return canvas
