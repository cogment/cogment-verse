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

import importlib

import gym
import numpy as np
from cogment_verse_environment.base import GymObservation
from cogment_verse_environment.env_spec import EnvSpec
from cogment_verse_environment.gym_env import GymEnv


def legal_moves_from_mask(action_mask):
    return [i for i, action in enumerate(action_mask) if action]


class PettingZooEnv(GymEnv):
    """
    Class for loading gym built-in environments.
    """

    def __init__(self, *, env_name, flatten=True, **kwargs):
        """
        Args:
            env_name: Name of the environment (NOTE: make sure it is available at gym.envs.registry.all())
        """
        self._flatten = flatten
        self._cumulative_rewards = None
        self._rewards = None
        super().__init__(env_name=env_name, **kwargs)
        self.num_players = len(self._env.action_spaces)

    def create_env(self, env_name, **_kwargs):
        """Function used to create the environment. Subclasses can override this method
        if they are using a gym style environment that needs special logic.
        """
        module_name = "pettingzoo.classic." + env_name
        env_module = importlib.import_module(module_name)
        self._env = env_module.env()

    def create_env_spec(self, env_name, **_kwargs):
        """Function used to create the specification. Subclasses can override this method
        if they are using a gym style ebservation, reward, done, self._turn, infonvironment that needs special logic.
        """

        action_space = self._env.action_spaces["player_0"]
        observation_space = self._env.observation_spaces["player_0"]["observation"]
        if isinstance(observation_space, gym.spaces.Tuple):
            obs_spaces = self._env.observation_space.spaces
        else:
            obs_spaces = [observation_space]
        if isinstance(action_space, gym.spaces.Tuple):
            act_spaces = action_space.spaces
        else:
            act_spaces = [action_space]

        act_dim = []
        for act_space in act_spaces:
            if isinstance(act_space, gym.spaces.Discrete):
                act_dim.append(act_space.n)
            else:
                act_dim.append(act_space.shape)

        return EnvSpec(
            env_name=env_name,
            obs_dim=[space.shape for space in obs_spaces],
            act_dim=act_dim,
            act_shape=[space.shape for space in act_spaces],
        )

    def _prepare_obs(self, obs):
        if self._flatten:
            return obs.reshape(-1)
        return obs

    def reset(self):
        self._env.reset()
        self._rewards = np.full(self.num_players, 0.0)
        self._cumulative_rewards = np.full(self.num_players, 0.0)
        self._turn = 0

        obs, _, done, info = self._env.last()

        return GymObservation(
            observation=self._prepare_obs(obs["observation"]),
            current_player=self._turn,
            legal_moves_as_int=legal_moves_from_mask(obs["action_mask"]),
            rewards=self._rewards,
            done=done,
            info=info,
        )

    def step(self, action=None):
        if isinstance(action, np.ndarray):
            action = action.tolist()

        self._env.step(action)
        obs, _, done, info = self._env.last()

        # 'last' method only returns reward for the actor that performed the action,
        # so for example we can lose the -1 reward for the losing player
        # NB: PettingZoo returns _cumulative rewards_,
        # so we have to compute the per-step reward
        cumulative_rewards = np.array([self._env.rewards[agent] for agent in self._env.agents])
        self._rewards = cumulative_rewards - self._cumulative_rewards
        self._cumulative_rewards = cumulative_rewards

        self._turn = (self._turn + 1) % self.num_players

        return GymObservation(
            observation=self._prepare_obs(obs["observation"]),
            current_player=self._turn,
            legal_moves_as_int=legal_moves_from_mask(obs["action_mask"]),
            rewards=self._rewards,
            done=done,
            info=info,
        )

    def render(self, mode="rgb_array"):
        return self._env.render(mode=mode)

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def close(self):
        self._env.close()
