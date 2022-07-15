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

from abc import ABC, abstractmethod
from collections import namedtuple

GymObservation = namedtuple(
    "GymObservation", ["observation", "current_player", "legal_moves_as_int", "rewards", "done", "info"]
)


class BaseEnv(ABC):
    def __init__(self, *, env_spec, num_players, framestack):
        self._env_spec = env_spec
        self.num_players = num_players
        self._turn = 0
        self._framestack = framestack
        assert framestack >= 1

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    def render(self, mode="rgb_array"):
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed=None):
        raise NotImplementedError

    def save(self, save_dir):
        raise NotImplementedError

    def load(self, load_dir):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    @property
    def env_spec(self):
        return self._env_spec

    @env_spec.setter
    def env_spec(self, env_spec):
        self._env_spec = env_spec
