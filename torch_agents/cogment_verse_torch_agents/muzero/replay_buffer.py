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
import torch
from collections import namedtuple


def clone_to_cpu(x):
    x = ensure_tensor(x)
    return x.detach().clone().cpu()


def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy())
    return torch.tensor(x)


EpisodeBatch = namedtuple(
    "EpisodeBatch",
    [
        "episode",
        "step",
        "state",
        "action",
        "target_reward",
        "next_state",
        "done",
        "target_policy",
        "target_value",
        "priority",
        "importance_weight",
        "target_value_probs",
        "target_reward_probs",
    ],
)


class Episode:
    def __init__(
        self, initial_state, discount, trial_id=0, min_priority=0.1, zero_reward_probs=None, zero_value_probs=None
    ):
        self._discount = discount
        self._id = trial_id
        self._states = [ensure_tensor(initial_state).clone().to("cpu")]
        self._actions = []
        self._rewards = []
        self._policy = []
        self._value = []
        self._done = []
        self._priority = []
        self._return = []
        self._reward_probs = []
        self._value_probs = []
        self._range = None
        self._bootstrap = None
        self._min_priority = min_priority
        self._action_space = set()
        self._zero_reward_probs = zero_reward_probs
        self._zero_value_probs = zero_value_probs

    def __len__(self):
        return len(self._actions)

    def total_priority(self):
        return sum(self._priority)

    def update_priority(self, step, priority):
        if step < len(self._priority):
            self._priority[step] = priority
            self._p = np.array(self._priority, dtype=np.double)
            self._p = self._p / self._p.sum()
            self._p = self._p.tolist()

    def bootstrap_value(self, steps, discount):
        length = len(self)
        self._bootstrap = [0.0 for _ in range(length)]
        for i in range(length):
            max_k = min(i + steps, length)
            if max_k < length:
                self._bootstrap[i] = (discount ** (max_k - i)) * self._value[max_k]

            for k in reversed(range(i, max_k)):
                self._bootstrap[i] += (discount ** (k - i)) * self._rewards[k]

    def add_step(self, state, action, reward_probs, reward, done, policy, value_probs, value):
        self._states.append(clone_to_cpu(state))
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._policy.append(clone_to_cpu(policy))
        self._value.append(float(value))
        self._done.append(int(done))
        self._reward_probs.append(clone_to_cpu(reward_probs))
        self._value_probs.append(clone_to_cpu(value_probs))
        self._p = None
        self._action_space.add(int(action))

        if done:
            N = len(self._actions)
            self._return = [0.0 for _ in range(N)]
            self._priority = [0.0 for _ in range(N)]
            self._return[-1] = reward
            for i in reversed(range(N - 1)):
                self._return[i] = self._rewards[i] + self._discount * self._return[i + 1]
                self._priority[i] = self._min_priority + abs(self._return[i] - self._value[i])
                # testing
                self._priority[i] = 1.0
            self._p = np.array(self._priority, dtype=np.double)
            self._p = self._p / self._p.sum()
            self._p = self._p.tolist()
            self._range = list(range(N))

    @property
    def done(self):
        return any(self._done)

    def sample(self, k) -> EpisodeBatch:
        start = np.random.randint(0, len(self._actions))
        end = start + k
        return self.episode_slice(start, end)

    def episode_slice(self, start, end) -> EpisodeBatch:
        assert end > start
        assert start < len(self._actions)
        length = end - start

        states = torch.zeros((length, *self._states[start].shape))
        actions = torch.zeros(length, dtype=torch.long)

        rewards = torch.zeros(length)
        done = torch.zeros(length)
        next_states = torch.zeros((length, *self._states[start].shape))
        target_policy = torch.zeros((length, *self._policy[start].shape))
        target_value = torch.zeros(length)
        priority = torch.zeros(length)
        importance_weight = torch.zeros(length)

        target_reward_probs = torch.zeros(length, *self._reward_probs[start].shape)
        target_value_probs = torch.zeros(length, *self._value_probs[start].shape)

        uniform_policy = torch.ones_like(ensure_tensor(self._policy[start]))
        uniform_policy /= torch.sum(uniform_policy)

        c = min(end, len(self._actions))

        for k in range(start, c):
            states[k - start] = ensure_tensor(self._states[k])
            next_states[k - start] = ensure_tensor(self._states[k + 1])
            actions[k - start] = self._actions[k]
            rewards[k - start] = self._rewards[k]
            target_policy[k - start] = ensure_tensor(self._policy[k])
            target_value[k - start] = self._bootstrap[k]
            priority[k - start] = self._p[k]
            importance_weight[k - start] = 0.0
            done[k - start] = self._done[k]

            target_reward_probs[k - start] = self._reward_probs[k]
            target_value_probs[k - start] = self._value_probs[k]

        for k in range(c, end):
            target_policy[k - start] = uniform_policy
            done[k - start] = 1
            actions[k - start] = np.random.randint(0, len(self._action_space))
            next_states[k - start] = next_states[k - start - 1]
            states[k - start] = next_states[k - start - 1]

            target_reward_probs[k - start] = self._zero_reward_probs
            target_value_probs[k - start] = self._zero_value_probs

        return EpisodeBatch(
            episode=torch.tensor(self._id),
            step=torch.tensor(start),
            state=states,
            action=actions,
            target_reward=rewards,
            next_state=next_states,
            done=done,
            target_policy=target_policy,
            target_value=target_value,
            priority=priority,
            importance_weight=importance_weight,
            target_reward_probs=target_reward_probs,
            target_value_probs=target_value_probs,
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
            self._priority[episode] = self._episodes[episode].total_priority()
        self._p = np.array(self._priority, dtype=np.double)
        self._p /= self._p.sum()

    def update_priorities(self, episodes, steps, priorities):
        modified_episodes = set()
        for episode, step, priority in zip(episodes, steps, priorities):
            modified_episodes.add(episode)
            self._episodes[episode].update_priority(step, priority)

        for episode in modified_episodes:
            self._priority[episode] = self._episodes[episode].total_priority()

        self._p = np.array(self._priority, dtype=np.double)
        self._p /= self._p.sum()

    def add_episode(self, episode):
        self._episodes.append(episode)
        self._total_size += len(episode)
        self._priority.append(episode.total_priority())

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

    def sample_old(self, rollout_length, batch_size) -> EpisodeBatch:
        # idx = np.random.randint(0, len(self._episodes), batch_size)
        idx = np.random.choice(self._range, batch_size, p=self._p)
        probs = [self._p[i] for i in idx]
        transitions = [self._episodes[i].sample(rollout_length) for i in idx]
        batch = [torch.stack(item) for item in zip(*transitions)]
        batch = EpisodeBatch(*batch)._asdict()
        batch["priority"] = batch["priority"][:, 0] * torch.tensor(probs)
        batch["importance_weight"] = 1 / (batch["priority"] + 1e-6) / self.size()

        return EpisodeBatch(**batch)

    def sample(self, rollout_length, batch_size) -> EpisodeBatch:
        prob = torch.tensor(list(map(len, self._episodes)), dtype=torch.double)
        prob /= torch.sum(prob)
        idx = torch.distributions.Categorical(prob).sample((batch_size,))
        transitions = [self._episodes[i].sample(rollout_length) for i in idx]
        items = []
        for item in zip(*transitions):
            item = torch.stack(item)
            if len(item.shape) >= 2:
                # (batch, rollout) -> (rollout, batch)
                item = torch.transpose(item, 0, 1)
            item = item.cpu().detach()
            items.append(item)

        batch = EpisodeBatch(*items)._asdict()
        batch["priority"] = torch.ones_like(batch["priority"])
        batch["importance_weight"] = torch.ones_like(batch["priority"])
        return EpisodeBatch(**batch)
