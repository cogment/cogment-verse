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

from cogment_verse_torch_agents.muzero.replay_buffer import Episode
from cogment_verse_torch_agents.muzero.networks import Distributional

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
def distribution():
    return Distributional(-10, 10, 32, 16)


@pytest.fixture
def zero_probs(distribution):
    return distribution.compute_target(torch.tensor(0.0, dtype=torch.float32))


def test_init(env, zero_probs):
    state = env.reset()
    episode = Episode(state, 0.99, zero_reward_probs=zero_probs, zero_value_probs=zero_probs)
    assert episode is not None


def test_add(env, policy, zero_probs, distribution):
    state = env.reset()
    episode = Episode(state, 0.99, zero_reward_probs=zero_probs, zero_value_probs=zero_probs)
    next_state, reward, done, _info = env.step(0)
    episode.add_step(
        next_state,
        0,
        distribution.compute_target(torch.tensor(reward)),
        reward,
        done,
        policy,
        zero_probs,
        0.0,
    )


def test_sample(env, policy, zero_probs, distribution):
    state = env.reset()
    episode = Episode(state, 0.99, zero_reward_probs=zero_probs, zero_value_probs=zero_probs)
    for _ in range(1000):
        next_state, reward, done, _info = env.step(0)
        episode.add_step(
            next_state,
            0,
            distribution.compute_target(torch.tensor(reward)),
            reward,
            done,
            policy,
            zero_probs,
            0.0,
        )
        if done:
            episode.bootstrap_value(10, 0.99)
            break

    assert episode.done[-1]
    # check that we can sample for small k
    batch = episode.sample(4)

    # check that we can sample when end padding is necessary
    length = len(episode) + 10
    batch = episode.sample(length)

    # check that stacking/concatenation is correct
    assert batch.state.shape == (length, 8)
    assert batch.action.shape == (length,)
    assert batch.target_reward.shape == (length,)
    assert batch.next_state.shape == batch.state.shape
    assert batch.target_policy.shape == (length, 4)
    assert batch.target_value.shape == (length,)
    assert torch.allclose(batch.state[1:], batch.next_state[:-1])
