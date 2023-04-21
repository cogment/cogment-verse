# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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
import torch

torch.multiprocessing.set_sharing_strategy("file_system")


class PPOReplayBufferSample:
    """PPO replay buffer's sample"""

    def __init__(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        adv: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        self.observation = observation
        self.action = action
        self.adv = adv
        self.value = value
        self.log_prob = log_prob

    def size(self) -> int:
        """get sample size"""
        return self.observation.size(dim=0)


class PPOReplayBuffer:
    """Replay buffer for PPO"""

    observations: torch.Tensor
    actions: torch.Tensor
    advs: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple,
        action_shape: tuple,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.dtype = dtype

        # Initialize data storage
        self.observations = torch.zeros((self.capacity, *self.observation_shape), dtype=self.dtype)
        self.actions = torch.zeros((self.capacity, *self.action_shape), dtype=self.dtype)
        self.advs = torch.zeros((self.capacity, 1), dtype=self.dtype)
        self.values = torch.zeros((self.capacity, 1), dtype=self.dtype)
        self.log_probs = torch.zeros((self.capacity, 1), dtype=self.dtype)
        self._ptr = 0
        self.num_total = 0
        self._rng = np.random.default_rng(seed)

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        adv: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        self.observations[self._ptr] = observation
        self.actions[self._ptr] = action
        self.advs[self._ptr] = adv
        self.values[self._ptr] = value
        self.log_probs[self._ptr] = log_prob
        self._ptr = (self._ptr + 1) % self.capacity
        self.num_total += 1

    def sample(self, num):
        size = self.size()
        if size < num:
            indices = range(size)
        else:
            indices = self._rng.choice(self.size(), size=num, replace=False)

        return PPOReplayBufferSample(
            observation=self.observations[indices],
            action=self.actions[indices],
            adv=self.advs[indices],
            value=self.values[indices],
            log_prob=self.log_probs[indices],
        )

    def size(self):
        return self.num_total if self.num_total < self.capacity else self.capacity


class TorchReplayBufferSample:
    def __init__(self, observation, next_observation, action, reward, done):
        self.observation = observation
        self.next_observation = next_observation
        self.action = action
        self.reward = reward
        self.done = done

    def size(self):
        return self.reward.size(dim=0)


class TorchReplayBuffer:
    def __init__(
        self,
        capacity,
        observation_shape,
        action_shape,
        seed=0,
        observation_dtype=torch.float,
        action_dtype=torch.float,
        reward_dtype=torch.float,
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.observation = torch.zeros((self.capacity, *self.observation_shape), dtype=observation_dtype)
        self.next_observation = torch.zeros((self.capacity, *self.observation_shape), dtype=observation_dtype)
        self.action = torch.zeros((self.capacity, *self.action_shape), dtype=action_dtype)
        self.reward = torch.zeros((self.capacity,), dtype=reward_dtype)
        self.done = torch.zeros((self.capacity,), dtype=torch.int8)

        self._ptr = 0
        self.num_total = 0

        self._rng = np.random.default_rng(seed)

    def add(self, observation, next_observation, action, reward, done):
        self.observation[self._ptr] = (
            observation.clone().detach() if torch.is_tensor(observation) else torch.tensor(observation)
        )
        self.next_observation[self._ptr] = (
            next_observation.clone().detach() if torch.is_tensor(next_observation) else torch.tensor(next_observation)
        )
        self.action[self._ptr] = action.clone().detach() if torch.is_tensor(action) else torch.tensor(action)
        self.reward[self._ptr] = reward.clone().detach() if torch.is_tensor(reward) else torch.tensor(reward)
        self.done[self._ptr] = done.clone().detach() if torch.is_tensor(done) else torch.tensor(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self.num_total += 1

    def sample(self, num):
        size = self.size()
        if size < num:
            indices = range(size)
        else:
            indices = self._rng.choice(self.size(), size=num, replace=False)

        return TorchReplayBufferSample(
            observation=self.observation[indices],
            next_observation=self.next_observation[indices],
            action=self.action[indices],
            reward=self.reward[indices],
            done=self.done[indices],
        )

    def size(self):
        return self.num_total if self.num_total < self.capacity else self.capacity
