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

from cogment_verse_environment.factory import make_environment
from cogment_verse_torch_agents.muzero.replay_buffer import Episode, TrialReplayBuffer

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def env():
    return make_environment("gym", "LunarLander-v2")


@pytest.fixture
def policy():
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def replay_buffer(env, policy):
    rb = TrialReplayBuffer(max_size=1000, discount_rate=0.99, bootstrap_steps=10)
    for _ in range(10):
        obs = env.reset()
        ep = Episode(obs.observation, 0.99)
        while not obs.done:
            action = 0
            next_obs = env.step(action)
            rb.add_sample(
                obs.observation, action, next_obs.rewards[0], next_obs.observation, next_obs.done, policy, 0.0
            )
            obs = next_obs

    return rb


def test_create(replay_buffer):
    assert replay_buffer.num_episodes() == 10
    assert replay_buffer.size() > 100


@pytest.mark.parametrize("rollout_length", [4, 8, 12])
def test_sample(replay_buffer, rollout_length):
    batch = replay_buffer.sample(rollout_length, 32)
    assert batch.state.shape == (32, rollout_length, 8)
    assert batch.next_state.shape == (32, rollout_length, 8)
    assert batch.action.shape == (32, rollout_length)
    assert batch.rewards.shape == (32, rollout_length)
    assert batch.target_policy.shape == (32, rollout_length, 4)
    assert batch.target_value.shape == (32, rollout_length)
    assert batch.done.shape == (32, rollout_length)
