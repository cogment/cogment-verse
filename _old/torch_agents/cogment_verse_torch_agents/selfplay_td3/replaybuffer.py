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

import numpy as np


class Memory:
    def __init__(self, **params):
        """Initialize buffer
        Params
        ======
            number_features (int): number of features in player's observation
            number_actions (int): number of actions of a player
            buffer_size (int): maximum size of the buffer
        """

        self.buffer_size = params["max_buffer_size"]
        self.state_dim = params["obs_dim1"]
        self.goal_dim = params["obs_dim2"]
        self.grid_shape = params["grid_shape"][0] * params["grid_shape"][1] * params["grid_shape"][2]
        self.act_dim = params["act_dim"]
        self.batch_size = params["batch_size"]

        # initialize state, action, reward, next_state, done arrays
        self._data = {}
        self._data["state"] = np.zeros((self.buffer_size, self.state_dim))
        self._data["goal"] = np.zeros((self.buffer_size, self.goal_dim))
        self._data["grid"] = np.zeros((self.buffer_size, self.grid_shape))
        self._data["action"] = np.zeros((self.buffer_size, self.act_dim))
        self._data["reward"] = np.zeros((self.buffer_size, 1))
        self._data["next_state"] = np.zeros((self.buffer_size, self.state_dim))
        self._data["next_goal"] = np.zeros((self.buffer_size, self.goal_dim))
        self._data["next_grid"] = np.zeros((self.buffer_size, self.grid_shape))
        self._data["player_done"] = np.zeros((self.buffer_size, 1))
        self._data["trial_done"] = np.zeros((self.buffer_size, 1))

        self._ptr = 0
        self._size = 0

    def add(self, data):
        """Add an experience to buffer
        Params
        ======
            data (list of tuples): (observation, action, reward, next_observation, done)
        """
        for sample in data:
            sample = sample._asdict()
            for _, key in enumerate(self._data):
                self._data[key][self._ptr, :] = sample[key]

            self._ptr = (self._ptr + 1) % self.buffer_size
            self._size = max(min(self._ptr, self.buffer_size), self._size)

    def sample(self):
        """Sample previous trajectory data from the buffer"""

        rval = {}
        sample_indices = np.random.randint(0, max(self._ptr, self._size), self.batch_size)
        for key, _ in self._data.items():
            rval[key] = self._data[key][sample_indices]

        return rval

    def get_size(self):
        return self._size
