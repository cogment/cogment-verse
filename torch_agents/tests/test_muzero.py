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

import os
from tempfile import TemporaryDirectory
import pytest
import torch

from lib.muzero import MuZeroMLP, Distributional
from lib.env.factory import make_environment
from third_party.hive.utils.schedule import LinearSchedule

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def rollout_length():
    return 3


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def env():
    return make_environment("gym", "LunarLander-v2")


@pytest.fixture
def obs_dim():
    return 8


@pytest.fixture
def act_dim():
    return 4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_create(obs_dim, act_dim, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    agent = MuZeroMLP(obs_dim=obs_dim, act_dim=act_dim, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_serialize(obs_dim, act_dim, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    agent = MuZeroMLP(obs_dim=obs_dim, act_dim=act_dim, device=device)

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "agent.dat")
        agent.save(filepath)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_deserialize(obs_dim, act_dim, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    agent = MuZeroMLP(obs_dim=obs_dim, act_dim=act_dim, device=device)

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "agent.dat")
        agent.save(filepath)

        params = agent._params
        print("PARAMS", params.keys())

        # check that we correctly change the NN architecture from the loaded params
        agent2 = MuZeroMLP(obs_dim=obs_dim + 1, act_dim=act_dim + 1, device="cpu")
        agent2.load(filepath)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_act(obs_dim, act_dim, env, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    agent = MuZeroMLP(obs_dim=obs_dim, act_dim=act_dim, epsilon_schedule=LinearSchedule(0.0, 0.0, 1), device=device)
    state = env.reset()

    for i in range(100):
        action = agent.act(state.observation, [0, 0, 0, 0])
        state = env.step(action)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_replay_buffer(obs_dim, act_dim, env, batch_size, rollout_length, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    agent = MuZeroMLP(
        obs_dim=obs_dim,
        act_dim=act_dim,
        epsilon_schedule=LinearSchedule(0.5, 0.0, 1000),
        mcts_depth=2,
        mcts_count=4,
        rollout_length=rollout_length,
        device=device,
    )
    state = env.reset()
    legal_moves = [1, 2, 3, 4]

    for episode in range(2):
        for step in range(10000):
            action = (episode + step) % act_dim
            next_state = env.step(action)
            sample = (
                state.observation,
                legal_moves,
                action,
                next_state.rewards,
                next_state.observation,
                legal_moves,
                next_state.done,
            )
            agent.consume_training_sample(sample)
            if next_state.done:
                state = env.reset()
                break
            state = next_state

    batch = agent.sample_training_batch(batch_size)
    assert batch["state"].shape == (batch_size, rollout_length, obs_dim)
    assert batch["action"].shape == (batch_size, rollout_length)
    assert batch["rewards"].shape == (batch_size, rollout_length, 1)
    assert batch["next_state"].shape == (batch_size, rollout_length, obs_dim)
    assert batch["target_policy"].shape == (batch_size, rollout_length, act_dim)
    assert batch["target_value"].shape == (batch_size, rollout_length)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("reanalyze_fraction", [0.0])  # , 1.0])
def test_learn(obs_dim, act_dim, env, batch_size, device, reanalyze_fraction):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    agent = MuZeroMLP(
        obs_dim=obs_dim,
        act_dim=act_dim,
        epsilon_schedule=LinearSchedule(0.5, 0.0, 1000),
        mcts_depth=2,
        mcts_count=4,
        rollout_length=3,
        device=device,
        reanalyze_fraction=reanalyze_fraction,
        reanalyze_period=1,
    )
    state = env.reset()
    legal_moves = [1, 2, 3, 4]

    for episode in range(2):
        for step in range(10000):
            action = (episode + step) % act_dim
            next_state = env.step(action)
            sample = (
                state.observation,
                legal_moves,
                action,
                next_state.rewards,
                next_state.observation,
                legal_moves,
                next_state.done,
            )
            agent.consume_training_sample(sample)
            if next_state.done:
                state = env.reset()
                break
            state = next_state

    batch = agent.sample_training_batch(batch_size)
    info = agent.learn(batch)


def test_distributional():
    dist = Distributional(-2.0, 3.0, 11)
    v = torch.tensor(1.738).to(torch.float32)
    t = dist.compute_target(v)
    assert torch.allclose(torch.sum(t * dist._bins), v)
    assert torch.sum(t != 0) == 2

    t = dist.compute_target(torch.tensor(-3.0, dtype=torch.float32))
    assert t[0] == 1
    assert t[1] == 0

    t = dist.compute_target(torch.tensor(4.0, dtype=torch.float32))
    assert t[-1] == 1
    assert t[-2] == 0
