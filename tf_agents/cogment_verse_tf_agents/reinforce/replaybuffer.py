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


class Memory():

    def __init__(self, number_features, number_actions, buffer_size):
        """Initialize buffer
        Params
        ======
            number_features (int): number of features in player's observation
            number_actions (int): number of actions of a player
            buffer_size (int): maximum size of the buffer
        """

        self.buffer_size = buffer_size
        self.number_features = number_features
        self.number_actions = number_actions

        # initialize state, action, reward, next_state, done arrays
        self._data = {}
        self._data['observations'] = np.zeros((self.buffer_size, self.number_features))
        self._data['actions'] = np.zeros((self.buffer_size,))
        self._data['rewards'] = np.zeros((self.buffer_size,))
        self._data['next_observations'] = np.zeros((self.buffer_size, self.number_features))
        self._data['dones'] = np.zeros((self.buffer_size,))

        self._ptr = 0

    def add(self, data):
        """Add an experience to buffer
        Params
        ======
            data (tuple): (observation, action, reward, next_observation, done)
        """
        for idx, key in enumerate(self._data):
            self._data[key][self._ptr] = np.asarray(data[idx])

        self._ptr = (self._ptr + 1) % self.buffer_size

    def sample(self):
        """Sample previous trajectory data from the buffer
        """

        rval = {}
        for key, _ in self._data.items():
            rval[key] = self._data[key][0:self._ptr]

        return rval

    def reset_replay_buffer(self):
        """
        Resets the pointer of the buffer to its initial position,
        thereby making previous trajectory data unavailable to the
        Reinforce agent
        """
        self._ptr = 0
