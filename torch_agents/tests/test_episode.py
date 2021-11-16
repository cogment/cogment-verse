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

from lib.env.factory import make_environment
from lib.replay_buffer import Episode

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def env():
    return make_environment("gym", "LunarLander-v2")


@pytest.fixture
def policy():
    return np.array([0.25, 0.25, 0.25, 0.25])


def test_init(env, policy):
    state = env.reset()
    episode = Episode(state)


def test_add(env):
    state = env.reset()
    episode = Episode(state.observation)
    next_state = env.step(0)
    episode.add_step(next_state.observation, 0, next_state.rewards, [next_state.done], policy, 0.0)


def test_sample(env, policy):
    state = env.reset()
    episode = Episode(state.observation)
    for _ in range(1000):
        next_state = env.step(0)
        episode.add_step(next_state.observation, 0, next_state.rewards, next_state.done, policy, 0.0)
        if episode.done:
            break

    assert episode.done
    # check that we can sample for small k
    batch = episode.sample(4)

    # check that we can sample when end padding is necessary
    N = len(episode) + 10
    batch = episode.sample(N)

    # check that stacking/concatenation is correct
    assert batch.state.shape == (N, 8)
    assert batch.action.shape == (N,)
    assert batch.rewards.shape == (N, 1)
    assert batch.next_state.shape == batch.state.shape
    assert batch.target_policy.shape == (N, 4)
    assert batch.target_value.shape == (N,)
    assert torch.allclose(batch.state[1:], batch.next_state[:-1])
