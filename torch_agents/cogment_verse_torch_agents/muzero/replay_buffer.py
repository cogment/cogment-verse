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
    ],
)


def pad_slice(lst, a, b, padval, dtype=torch.float32):
    if b >= len(lst):
        lst = lst + [padval] * (b - len(lst) + 1)
    # assert type(lst[-1]) == type(padval), f"{type(lst[-1])} == {type(padval)}"
    if type(padval) == torch.Tensor:
        # assert padval.shape == lst[-1].shape
        lst = [t if t.shape else t.unsqueeze(0) for t in lst]
        return torch.clone(torch.cat(lst[a:b]).type(dtype))
    else:
        # array of scalars
        return torch.clone(torch.tensor(lst[a:b]).type(dtype))


class Episode:
    def __init__(self, initial_state, discount, id=0, min_priority=100.0):
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
        # assert not self._done

        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._policy.append(policy)
        self._value.append(value)
        self._done.append(done)
        self._p = None

        if done:
            N = len(self._actions)
            self._return = [0.0 for _ in range(N)]
            self._priority = [0.0 for _ in range(N)]
            self._return[-1] = reward
            for i in reversed(range(N - 1)):
                self._return[i] = self._rewards[i] + self._discount * self._return[i + 1]
                self._priority[i] = min(self._min_priority, abs(self._return[i] - self._value[i]))
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
        if isinstance(self._rewards[-1], (np.ndarray, torch.Tensor, float)):
            null_reward = 0.0 * self._rewards[-1]
        else:
            null_reward = [0.0 for _ in self._rewards[-1]]
        return EpisodeBatch(
            torch.tensor(self._id),
            torch.tensor(a),
            pad_slice(self._states, a, b, self._states[-1]),
            pad_slice(self._actions, a, b, self._actions[-1], dtype=torch.long),
            pad_slice(self._rewards, a, b, null_reward),
            pad_slice(self._states, a + 1, b + 1, self._states[-1]),
            pad_slice(self._done, a, b, self._done[-1]),
            pad_slice(self._policy, a, b, self._policy[-1]),
            # pad_slice(self._value, a, b, 0.0),
            # pad_slice(self._return, a, b, 0.0),
            pad_slice(self._bootstrap, a, b, 0.0),
            pad_slice(self._p, a, b, 0.0),
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
        for i, prob in enumerate(probs):
            batch["priority"][i] = batch["priority"][i] * probs[i]

        return batch

    def add_sample(self, sample):
        state, _, action, reward, next_state, _, done, policy, value = sample  # Hive sample producer

        if self._current_episode is None:
            self._current_episode = Episode(state, self._discount_rate, id=len(self._episodes))

        self._current_episode.add_step(next_state, action, reward, done, policy, value)

        if done:
            self._current_episode.bootstrap_value(self._bootstrap_steps, self._discount_rate)
            self._add_episode(self._current_episode)
            self._current_episode = None


def replay_buffer_worker(
    *,
    discount_rate,
    bootstrap_steps,
    sample_queue,
    batch_queue,
    update_queue,
    batch_size,
    min_size,
    max_size,
    rollout_length,
    total_size,
    terminate,
):
    replay_buffer = TrialReplayBuffer(max_size, discount_rate, bootstrap_steps)

    while not terminate.value:
        # apply all pending priority updates
        while not update_queue.empty():
            try:
                updates = update_queue.get_nowait()
                for (episode, step, priority) in updates:
                    replay_buffer.update_priority(episode, step, priority)
                # print("RB UPDATED_PRIORITY")
            except queue.Empty:
                break

        # wait for next sample
        while not sample_queue.empty():
            try:
                sample = sample_queue.get_nowait()
                replay_buffer.add_sample(sample)
                # print("RB ADDED SAMPLE")
            except queue.Empty:
                continue

        total_size.value = replay_buffer.size()
        # print("RB SIZE", total_size.value)

        if replay_buffer.size() < min_size:
            continue

        # push next training batch
        batch = replay_buffer.sample(rollout_length, batch_size)
        while True:
            try:
                batch_queue.put(batch, timeout=1.0)
                break
            except queue.Full:
                continue

        # print("RB ADDED TRAINING BATCH", batch_queue.qsize(), total_size.value)


class ConcurrentTrialReplayBuffer(mp.Process):
    def __init__(self, *, batch_size, bootstrap_steps, min_size, max_size, rollout_length, discount_rate):
        super().__init__()
        sample_qsize = 10000
        batch_qsize = 8
        update_qsize = 10000
        ctx = mp  # .get_context("spawn")
        self._sample_queue = ctx.Queue(sample_qsize)
        self._batch_queue = ctx.Queue(batch_qsize)
        self._update_queue = ctx.Queue(update_qsize)
        self._terminate = ctx.Value(ctypes.c_bool, False)
        self._bootstrap_steps = bootstrap_steps
        self._total_size = ctx.Value(ctypes.c_int64, 0)

        self._kwargs = {
            "sample_queue": self._sample_queue,
            "batch_queue": self._batch_queue,
            "update_queue": self._update_queue,
            "batch_size": batch_size,
            "min_size": min_size,
            "max_size": max_size,
            "rollout_length": rollout_length,
            "discount_rate": discount_rate,
            "rollout_length": rollout_length,
            "terminate": self._terminate,
            "bootstrap_steps": self._bootstrap_steps,
            "total_size": self._total_size,
        }

    def run(self):
        replay_buffer_worker(**self._kwargs)

    def update_priorities(self, updates):
        while True:
            try:
                self._update_queue.put(updates, timeout=1.0)
                break
            except queue.Full:
                continue

    def add_sample(self, sample):
        self._sample_queue.put(sample)

    def sample(self, rollout_length, batch_size):
        # print("TRAINING BATCH REQUESTED")
        while True:
            try:
                batch = self._batch_queue.get(timeout=1.0)
                # print("RECEIVED BATCH")
                return batch
            except queue.Empty:
                # print("NO TRAINING BATCH IN QUEUE")
                continue

    def size(self):
        # print("ConcurrentReplayBuffer::size(): RB total size", self._total_size.value)
        return self._total_size.value
