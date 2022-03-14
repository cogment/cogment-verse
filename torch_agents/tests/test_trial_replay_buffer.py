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

import pytest
import numpy as np
import torch

from cogment_verse_torch_agents.muzero.replay_buffer import Episode, TrialReplayBuffer

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def env():
    class MockEnvironment:
        def __init__(self):
            self.num_action = 4
            self.num_input = 8

        def reset(self):
            return np.random.rand(self.num_input)

        def step(self, _action):
            done = np.random.rand() < 0.2
            return np.random.rand(self.num_input), np.random.rand(), done, {}

    return MockEnvironment()


@pytest.fixture
def policy():
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def replay_buffer(env, policy):
    reward_probs = torch.ones(4) / 4
    value_probs = torch.ones(4) / 4
    replay_buffer = TrialReplayBuffer(max_size=1000)
    for _ in range(10):
        obs = env.reset()
        done = False
        episode = Episode(obs, 0.99, zero_reward_probs=reward_probs, zero_value_probs=value_probs)

        while not done:
            action = 0
            next_obs, reward, done, _info = env.step(action)
            episode.add_step(next_obs, action, reward_probs, reward, done, policy, value_probs, 0.0)
            obs = next_obs

        episode.bootstrap_value(100, 0.99)
        replay_buffer.update_episode(episode)

    return replay_buffer


def test_create(replay_buffer):
    assert replay_buffer.num_episodes() == 10
    assert replay_buffer.size() == sum(len(episode) for episode in replay_buffer.episodes.values())


@pytest.mark.parametrize("rollout_length", [4, 8, 12])
def test_sample(replay_buffer, rollout_length):
    batch = replay_buffer.sample(rollout_length, 32)
    assert batch.state.shape == (rollout_length, 32, 8)
    assert batch.next_state.shape == (rollout_length, 32, 8)
    assert batch.action.shape == (rollout_length, 32)
    assert batch.target_reward.shape == (rollout_length, 32)
    assert batch.target_policy.shape == (rollout_length, 32, 4)
    assert batch.target_value.shape == (rollout_length, 32)
    assert batch.done.shape == (rollout_length, 32)
