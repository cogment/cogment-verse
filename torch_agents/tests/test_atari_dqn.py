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
import pytest
import torch
from cogment_verse_torch_agents.atari_cnn import NatureAtariDQNModel
from cogment_verse_torch_agents.wrapper import format_legal_moves

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.mark.parametrize("framestack", [1, 2, 4])
@pytest.mark.parametrize("act_dim", [5])
@pytest.mark.parametrize("seed", [42, 56, 78, 10967])
def test_cnn_action(act_dim, framestack, seed):
    rng = np.random.default_rng(seed)
    legal_moves_as_int = []

    dqn = NatureAtariDQNModel(obs_dim=84 * 84, act_dim=act_dim, framestack=framestack, seed=seed)

    random_observation = rng.random((framestack, 84, 84))
    assert random_observation.shape == (framestack, 84, 84)

    action = dqn.act(random_observation, format_legal_moves(legal_moves_as_int, act_dim))
    assert 0 <= action < act_dim


@pytest.mark.parametrize("act_dim", [5])
@pytest.mark.parametrize("seed", [42, 56, 78, 10967])
def test_cnn_learn(act_dim, seed):
    torch.autograd.set_detect_anomaly(True)
    rng = np.random.default_rng(seed)
    framestack = 1
    legal_moves_as_int = []
    batch_size = 4

    dqn = NatureAtariDQNModel(obs_dim=84 * 84, act_dim=act_dim, framestack=framestack, seed=seed)

    batch = {
        "observations": rng.random((batch_size, framestack, 84, 84), dtype=np.float32),
        "next_observations": rng.random((batch_size, framestack, 84, 84), dtype=np.float32),
        "rewards": rng.random((batch_size, 1), dtype=np.float32),
        "actions": rng.integers(0, high=act_dim, size=batch_size).reshape(-1, 1),
        "legal_moves_as_int": np.array([format_legal_moves(legal_moves_as_int, act_dim)] * batch_size),
        "done": rng.integers(0, high=2, size=batch_size).reshape(-1, 1),
    }

    info = dqn.learn(batch)
    assert "loss" in info


@pytest.mark.parametrize("act_dim", [5])
@pytest.mark.parametrize("seed", [42, 56, 78, 10967])
def test_replay_buffer(act_dim, seed):
    torch.autograd.set_detect_anomaly(True)
    rng = np.random.default_rng(seed)
    framestack = 1
    legal_moves_as_int = []

    dqn = NatureAtariDQNModel(obs_dim=84 * 84, act_dim=act_dim, framestack=framestack, seed=seed)

    for _ in range(32):
        obs = rng.random((framestack, 84, 84))
        legal_moves = format_legal_moves(legal_moves_as_int, act_dim)
        action = dqn.act(obs, legal_moves)
        next_obs = rng.random((framestack, 84, 84), dtype=np.float32)
        rewards = rng.random(1, dtype=np.float32)
        done = rng.integers(0, high=2)

        data = (
            obs.astype(np.float32),
            np.array(legal_moves),
            action,
            rewards,
            next_obs,
            np.array(legal_moves),
            done,
        )

        dqn.consume_training_sample(data)

        obs = next_obs

    assert dqn.replay_buffer_size() == 32
    batch = dqn.sample_training_batch(32)
    info = dqn.learn(batch)
    assert "loss" in info
