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

import ctypes
import queue
import numpy as np
import torch
import queue
from collections import namedtuple
import torch.multiprocessing as mp

EpisodeBatch = namedtuple(
    "EpisodeBatch",
    [
        "episode",
        "step",
        "state",
        "action",
        "rewards",
        "next_state",
        "done",
        "target_policy",
        "target_value",
        "priority",
        "importance_weight",
    ],
)


class Episode:
    def __init__(self, initial_state, discount, id=0, min_priority=0.1):
        self._discount = discount
        self._id = 0
        self._states = [initial_state]
        self._actions = []
        self._rewards = []
        self._policy = []
        self._value = []
        self._done = []
        self._priority = []
        self._return = []
        self._range = None
        self._bootstrap = None
        self._min_priority = min_priority
        self._action_space = set()

    def __len__(self):
        return len(self._actions)

    def update_priority(self, step, priority):
        if step < len(self._priority):
            self._priority[step] = priority
            self._p = np.array(self._priority, dtype=np.double)
            self._p = self._p / self._p.sum()
            self._p = self._p.tolist()

    def bootstrap_value(self, steps, discount):
        N = len(self)
        self._bootstrap = [0.0 for _ in range(N)]
        for i in range(N):
            max_k = min(i + steps, N)
            if max_k < N:
                self._bootstrap[i] = (discount ** (max_k - i)) * self._value[max_k]

            for k in reversed(range(i, max_k)):
                self._bootstrap[i] += (discount ** (k - i)) * self._rewards[k]

    def add_step(self, state, action, reward, done, policy, value):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._policy.append(policy)
        self._value.append(value)
        self._done.append(done)
        self._p = None
        self._action_space.add(action)

        if done:
            N = len(self._actions)
            self._return = [0.0 for _ in range(N)]
            self._priority = [0.0 for _ in range(N)]
            self._return[-1] = reward
            for i in reversed(range(N - 1)):
                self._return[i] = self._rewards[i] + self._discount * self._return[i + 1]
                self._priority[i] = self._min_priority + abs(self._return[i] - self._value[i])
            self._p = np.array(self._priority, dtype=np.double)
            self._p = self._p / self._p.sum()
            self._p = self._p.tolist()
            self._range = list(range(N))

    @property
    def done(self):
        return any(self._done)

    def sample(self, k) -> EpisodeBatch:
        a = np.random.choice(self._range, 1, p=self._p).item()
        b = a + k
        return self.episode_slice(a, b)

    def episode_slice(self, a, b) -> EpisodeBatch:
        assert b > a
        assert a < len(self._actions)

        states = torch.zeros((b - a, *self._states[a].shape), dtype=self._states[a].dtype)
        actions = torch.zeros((b - a), dtype=torch.long)

        rewards = torch.zeros((b - a), dtype=self._states[a].dtype)
        done = torch.zeros((b - a), dtype=torch.long)
        next_states = torch.zeros((b - a, *self._states[a].shape), dtype=self._states[a].dtype)
        target_policy = torch.zeros((b - a, *self._policy[a].shape), dtype=self._states[a].dtype)
        target_value = torch.zeros((b - a,), dtype=self._states[a].dtype)
        priority = torch.zeros((b - a,), dtype=self._states[a].dtype)
        importance_weight = torch.zeros((b - a,), dtype=self._states[a].dtype)

        uniform_policy = torch.ones_like(torch.tensor(self._policy[a]))
        uniform_policy /= torch.sum(uniform_policy)

        c = min(b, len(self._actions))

        for k in range(a, c):
            states[k - a] = torch.tensor(self._states[k])
            next_states[k - a] = torch.tensor(self._states[k + 1])
            actions[k - a] = self._actions[k]
            rewards[k - a] = self._rewards[k]
            target_policy[k - a] = torch.from_numpy(self._policy[k])
            target_value[k - a] = self._bootstrap[k]
            priority[k - a] = self._p[k]
            importance_weight[k - a] = 0.0
            done[k - a] = self._done[k]

        for k in range(c, b):
            target_policy[k - a] = uniform_policy
            done[k - a] = 1
            actions[k - a] = np.random.randint(0, len(self._action_space))
            next_states[k - a] = next_states[k - a - 1]
            states[k - a] = next_states[k - a - 1]

        return EpisodeBatch(
            torch.tensor(self._id),
            torch.tensor(a),
            states,
            actions,
            rewards,
            next_states,
            done,
            target_policy,
            target_value,
            priority,
            importance_weight,
        )


class TrialReplayBuffer:
    def __init__(self, max_size, discount_rate, bootstrap_steps):
        self._episodes = []
        self._max_size = max_size
        self._total_size = 0
        self._priority = []
        self._discount_rate = discount_rate
        self._bootstrap_steps = bootstrap_steps
        self._p = None
        self._range = None
        self._current_episode = None
        self._bootstrap_steps = bootstrap_steps

    def size(self):
        return self._total_size

    def num_episodes(self):
        return len(self._episodes)

    def update_priority(self, episode, step, priority):
        if episode < len(self._episodes):
            self._episodes[episode].update_priority(step, priority)
            self._priority[episode] = sum(self._episodes[episode]._priority)
        self._p = np.array(self._priority, dtype=np.double)
        self._p /= self._p.sum()

    def update_priorities(self, episodes, steps, priorities):
        modified_episodes = set()
        for episode, step, priority in zip(episodes, steps, priorities):
            modified_episodes.add(episode)
            self._episodes[episode].update_priority(step, priority)

        for episode in modified_episodes:
            self._priority[episode] = sum(self._episodes[episode]._priority)

        self._p = np.array(self._priority, dtype=np.double)
        self._p /= self._p.sum()

    def _add_episode(self, episode):
        self._episodes.append(episode)
        self._total_size += len(episode)
        self._priority.append(sum(episode._priority))

        # discard old episodes if necessary
        overflow = self._total_size - self._max_size
        discarded = 0
        idx = 0
        while discarded < overflow:
            discarded += len(self._episodes[idx])
            idx += 1

        if discarded > 0:
            self._total_size -= discarded
            self._episodes = self._episodes[idx:]
            self._priority = self._priority[idx:]

        self._p = np.array(self._priority, dtype=np.double)
        self._p /= self._p.sum()
        self._range = list(range(len(self._episodes)))

    def sample(self, rollout_length, batch_size) -> EpisodeBatch:
        # idx = np.random.randint(0, len(self._episodes), batch_size)
        idx = np.random.choice(self._range, batch_size, p=self._p)
        probs = [self._p[i] for i in idx]
        transitions = [self._episodes[i].sample(rollout_length) for i in idx]
        batch = [torch.stack(item) for item in zip(*transitions)]
        batch = EpisodeBatch(*batch)._asdict()
        batch["priority"] = batch["priority"][:, 0] * torch.tensor(probs)
        batch["importance_weight"] = 1 / (batch["priority"] + 1e-6) / self.size()

        return EpisodeBatch(**batch)

    def add_sample(self, state, action, reward, next_state, done, policy, value):
        if self._current_episode is None:
            self._current_episode = Episode(state, self._discount_rate, id=len(self._episodes))

        self._current_episode.add_step(next_state, action, reward, done, policy, value)

        if done:
            self._current_episode.bootstrap_value(self._bootstrap_steps, self._discount_rate)
            self._add_episode(self._current_episode)
            self._current_episode = None
