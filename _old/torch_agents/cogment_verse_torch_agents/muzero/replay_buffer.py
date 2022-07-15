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

from collections import namedtuple, OrderedDict
import copy
import time
import numpy as np
import torch


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
        "target_value_probs",
        "target_reward_probs",
    ],
)


class Episode:
    def __init__(self, initial_state, discount, trial_id=0, zero_reward_probs=None, zero_value_probs=None):
        self._discount = discount
        self._id = trial_id
        self.states = [clone_to_cpu(initial_state)]
        self.actions = []
        self.rewards = []
        self._policy = []
        self._value = []
        self.done = []
        self._return = []
        self._reward_probs = []
        self._value_probs = []
        self.bootstrap = None
        self._action_space = set()
        self.zero_reward_probs = clone_to_cpu(zero_reward_probs)
        self.zero_value_probs = clone_to_cpu(zero_value_probs)
        self.timestamp = time.time()

    def clone(self):
        episode = Episode(self.states[0], self._discount, self._id, self.zero_reward_probs, self.zero_value_probs)
        for step in range(len(self)):
            episode.add_step(
                self.states[step + 1],
                self.actions[step],
                self._reward_probs[step],
                self.rewards[step],
                self.done[step],
                self._policy[step],
                self._value_probs[step],
                self._value[step],
            )
            episode.bootstrap = copy.deepcopy(self.bootstrap)
        return episode

    def __len__(self):
        return len(self.actions)

    @torch.no_grad()
    def bootstrap_value(self, steps, discount):
        length = len(self)
        self.bootstrap = [0.0 for _ in range(length)]
        for i in range(length):
            max_k = min(i + steps, length)
            if max_k < length:
                self.bootstrap[i] = (discount ** (max_k - i)) * self._value[max_k]

            for k in reversed(range(i, max_k)):
                self.bootstrap[i] += (discount ** (k - i)) * self.rewards[k]

    def add_step(self, state, action, reward_probs, reward, done, policy, value_probs, value):
        self.states.append(clone_to_cpu(state))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self._policy.append(clone_to_cpu(policy))
        self._value.append(float(value))
        self.done.append(int(done))
        self._reward_probs.append(clone_to_cpu(reward_probs))
        self._value_probs.append(clone_to_cpu(value_probs))
        self._action_space.add(int(action))

        if done:
            num_steps = len(self.actions)
            self._return = [0.0 for _ in range(num_steps)]
            self._return[-1] = reward
            for i in reversed(range(num_steps - 1)):
                self._return[i] = self.rewards[i] + self._discount * self._return[i + 1]

    def sample(self, k) -> EpisodeBatch:
        start = np.random.randint(0, len(self.actions))
        end = start + k
        return self.episode_slice(start, end)

    def episode_slice(self, start, end) -> EpisodeBatch:
        assert end > start
        assert start < len(self.actions)
        length = end - start

        states = torch.zeros((length, *self.states[start].shape))
        actions = torch.zeros(length, dtype=torch.long)

        rewards = torch.zeros(length)
        done = torch.zeros(length)
        next_states = torch.zeros((length, *self.states[start].shape))
        target_policy = torch.zeros((length, *self._policy[start].shape))
        target_value = torch.zeros(length)

        target_reward_probs = torch.zeros(length, *self._reward_probs[start].shape)
        target_value_probs = torch.zeros(length, *self._value_probs[start].shape)

        uniform_policy = torch.ones_like(ensure_tensor(self._policy[start]))
        uniform_policy /= torch.sum(uniform_policy)

        c = min(end, len(self.actions))

        for k in range(start, c):
            states[k - start] = ensure_tensor(self.states[k])
            next_states[k - start] = ensure_tensor(self.states[k + 1])
            actions[k - start] = self.actions[k]
            rewards[k - start] = self.rewards[k]
            target_policy[k - start] = ensure_tensor(self._policy[k])
            target_value[k - start] = self.bootstrap[k]
            done[k - start] = self.done[k]

            target_reward_probs[k - start] = self._reward_probs[k]
            target_value_probs[k - start] = self._value_probs[k]

        for k in range(c, end):
            target_policy[k - start] = uniform_policy
            done[k - start] = 1
            actions[k - start] = np.random.randint(0, len(self._action_space))
            next_states[k - start] = next_states[k - start - 1]
            states[k - start] = next_states[k - start - 1]

            target_reward_probs[k - start] = self.zero_reward_probs
            target_value_probs[k - start] = self.zero_value_probs

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
            target_reward_probs=target_reward_probs,
            target_value_probs=target_value_probs,
        )


class TrialReplayBuffer:
    def __init__(self, max_size):
        self.episodes = OrderedDict()
        self._max_size = max_size
        self._total_size = 0
        self._p = None
        self._current_episode = None

    def size(self):
        return self._total_size

    def num_episodes(self):
        return len(self.episodes)

    def update_episode(self, episode, key=None):
        if key is None:
            key = len(self.episodes)

        self.episodes[key] = episode
        self._total_size = sum(map(len, self.episodes.values()))

        # discard old episodes if necessary
        self._discard_old_episodes()

    def _discard_old_episodes(self):
        keys_to_delete = []
        for key, episode in self.episodes.items():
            if self._total_size < self._max_size:
                break
            self._total_size -= len(episode)
            keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.episodes[key]

    def sample(self, rollout_length, batch_size) -> EpisodeBatch:
        keys = list(self.episodes.keys())
        lengths = [len(episode) for _, episode in self.episodes.items()]
        prob = torch.tensor(lengths, dtype=torch.double)
        prob /= torch.sum(prob)
        idx = torch.distributions.Categorical(prob).sample((batch_size,))
        transitions = [self.episodes[keys[i]].sample(rollout_length) for i in idx]
        items = []
        for item in zip(*transitions):
            item = torch.stack(item)
            if len(item.shape) >= 2:
                # (batch, rollout) -> (rollout, batch)
                item = torch.transpose(item, 0, 1)
            item = item.cpu().detach()
            items.append(item)

        batch = EpisodeBatch(*items)._asdict()
        return EpisodeBatch(**batch)
