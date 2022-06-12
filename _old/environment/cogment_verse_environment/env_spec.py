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

# No imports


class EnvSpec:
    def __init__(self, env_name, obs_dim, act_dim, act_shape, env_info=None):
        self._env_name = env_name
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._act_shape = act_shape
        self._env_info = {} if env_info is None else env_info

    @property
    def env_name(self):
        return self._env_name

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def act_dim(self):
        return self._act_dim

    @property
    def env_info(self):
        return self._env_info

    @property
    def act_shape(self):
        return self._act_shape
