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

import gym
from cogment_verse_environment.base import BaseEnv, GymObservation
from cogment_verse_environment.env_spec import EnvSpec


class GymEnv(BaseEnv):
    """
    Class for loading gym built-in environments.
    """

    def __init__(self, *, env_name, num_players=1, framestack=1, **kwargs):
        """
        Args:
            env_name: Name of the environment (NOTE: make sure it is available at gym.envs.registry.all())
        """
        self.create_env(env_name, **kwargs)
        super().__init__(
            env_spec=self.create_env_spec(env_name, **kwargs), num_players=num_players, framestack=framestack
        )

    def create_env(self, env_name, **_kwargs):
        """Function used to create the environment. Subclasses can override this method
        if they are using a gym style environment that needs special logic.
        """
        self._env = gym.make(env_name)

    def create_env_spec(self, env_name, **_kwargs):
        """Function used to create the specification. Subclasses can override this method
        if they are using a gym style environment that needs special logic.
        """
        if isinstance(self._env.observation_space, gym.spaces.Tuple):
            obs_spaces = self._env.observation_space.spaces
        else:
            obs_spaces = [self._env.observation_space]
        if isinstance(self._env.action_space, gym.spaces.Tuple):
            act_spaces = self._env.action_space.spaces
        else:
            act_spaces = [self._env.action_space]

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

    def reset(self):
        observation = self._env.reset()
        return GymObservation(
            observation=observation,
            rewards=[0.0],
            current_player=self._turn,
            legal_moves_as_int=[],
            done=False,
            info={},
        )

    def step(self, action=None):
        observation, reward, done, info = self._env.step(action)
        self._turn = (self._turn + 1) % self.num_players
        return GymObservation(
            observation=observation,
            current_player=self._turn,
            legal_moves_as_int=[],
            rewards=[reward],
            done=done,
            info=info,
        )

    def render(self, mode="rgb_array"):
        return self._env.render(mode=mode)

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def close(self):
        self._env.close()
