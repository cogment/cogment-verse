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
        device: torch.device,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.dtype = dtype
        self.device = device
        self.seed = seed

        # Initialize data storage
        self.observations = torch.zeros((self.capacity, *self.observation_shape), dtype=self.dtype)
        self.actions = torch.zeros((self.capacity, *self.action_shape), dtype=self.dtype)
        self.advs = torch.zeros((self.capacity, 1), dtype=self.dtype)
        self.values = torch.zeros((self.capacity, 1), dtype=self.dtype)
        self.log_probs = torch.zeros((self.capacity, 1), dtype=self.dtype)
        self._ptr = 0
        self.num_total = 0
        self.count = 0

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        adv: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        self.observations[self._ptr] = observation
        self.actions[self._ptr] = action
        self.advs[self._ptr] = adv
        self.values[self._ptr] = value
        self.log_probs[self._ptr] = log_prob
        self._ptr = (self._ptr + 1) % self.capacity
        self.num_total += 1

    def add_multi_samples(
        self, trial_obs: list, trial_act: list, trial_adv: list, trial_val: list, trial_log_prob: list
    ) -> None:
        for obs, act, adv, val, log_prob in zip(trial_obs, trial_act, trial_adv, trial_val, trial_log_prob):
            self.add(observation=obs, action=act, adv=adv, value=val, log_prob=log_prob)

        self.count += 1

    def sample(self, num) -> PPOReplayBufferSample:
        np.random.seed(self.seed + self.count)
        size = self.size()
        if size < num:
            indices = range(size)
        else:
            indices = np.random.choice(self.size(), size=num, replace=False)
        return PPOReplayBufferSample(
            observation=self.observations[indices].clone().to(self.device),
            action=self.actions[indices].clone().to(self.device),
            adv=self.advs[indices].clone().to(self.device),
            value=self.values[indices].clone().to(self.device),
            log_prob=self.log_probs[indices].clone().to(self.device),
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
            observation=self.observation[indices].clone(),
            next_observation=self.next_observation[indices].clone(),
            action=self.action[indices].clone(),
            reward=self.reward[indices].clone(),
            done=self.done[indices].clone(),
        )

    def size(self):
        return self.num_total if self.num_total < self.capacity else self.capacity
