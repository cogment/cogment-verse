# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import os
from typing import Tuple, Union

import numpy as np
from numpy import savez_compressed


class HumanDataBuffer:
    """Storage demonstration data and feedback frmm experts where each observation correspond
    to an action"""

    def __init__(
        self,
        observation_shape: tuple,
        action_shape: tuple,
        file_name: str = "",
        saving_path: str = "./cogment_verse/replay_buffers/data",
        seed: int = 0,
        capacity: int = 100_000,
        observation_dtype=np.float32,
        action_dtype=np.float32,
        human_data_category: str = "demo",
        loading: bool = False,
        saving_iter: int = 3,
    ) -> None:
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.file_name = file_name
        self.saving_path = saving_path
        self.seed = seed
        self.capacity = capacity
        self.observation_dtype = observation_dtype
        self.action_dtype = action_dtype
        self.human_data_category = human_data_category
        self.loading = loading
        self.saving_iter = saving_iter

    @property
    def loading(self) -> bool:
        """Get loading options"""
        return self._loading

    @loading.setter
    def loading(self, value: bool) -> None:
        """Set loading option"""
        self._loading = value
        if self._loading:
            self.load_buffer()
        else:
            self.observations = np.zeros((self.capacity, *self.observation_shape), dtype=self.observation_dtype)
            self.actions = np.zeros((self.capacity, *self.action_shape), dtype=self.action_dtype)
            if self.human_data_category == "feedback":
                self.feedback = np.zeros((self.capacity, 1), dtype=np.float32)
            self._ptr = 0
            self.num_total = 0
            self._rng = np.random.default_rng(self.seed)

    def add(self, observation: np.ndarray, action: np.ndarray, feedback: Union[np.ndarray, None] = None) -> None:
        """Add new sample to buffer"""

        self.observations[self._ptr] = observation
        self.actions[self._ptr] = action
        if self.human_data_category == "feedback":
            self.feedback[self._ptr] = feedback

        self._ptr = (self._ptr + 1) % self.capacity
        self.num_total += 1
        if self.num_total % self.saving_iter == 0:
            self.save_buffer()

    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Select randomly the samples"""
        if self.num_total < num_samples:
            indices = range(self.num_total)
        else:
            indices = self._rng.choice(self.num_total, size=num_samples, replace=False)

        return (self.observations[indices], self.actions[indices])

    def save_buffer(self) -> None:
        """Save human demonstration in numpy binary array *npz"""
        os.makedirs(self.saving_path, exist_ok=True)
        file_path = f"{self.saving_path}/{self.file_name}.npz"
        if self.human_data_category == "demo":
            savez_compressed(file_path, observations=self.observations[: self._ptr], actions=self.actions[: self._ptr])
        elif self.human_data_category == "feedback":
            savez_compressed(
                file_path,
                observations=self.observations[: self._ptr],
                actions=self.actions[: self._ptr],
                feedback=self.feedback[: self._ptr],
            )
        else:
            raise ValueError("Human data category does not exist. Can be either demo or feedback.a")

    def load_buffer(self) -> None:
        """Load numpy binary file"""
        file_path = f"{self.saving_path}/{self.file_name}.npz"
        loaded_data = np.load(file_path)
        self.observations = loaded_data["observations"]
        self.actions = loaded_data["actions"]
        if self.human_data_category == "feedback":
            self.feedback = loaded_data["feedback"]
        self.num_total = self.observations.shape[0]
        self._ptr = self.num_total
