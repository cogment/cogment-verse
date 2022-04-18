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

import copy
import time
import pytest
import torch
import torch.multiprocessing as mp
import numpy as np

from cogment_verse_torch_agents.muzero.networks import (
    MuZero,
    Distributional,
    reward_transform,
    reward_transform_inverse,
)

from cogment_verse_torch_agents.muzero.agent import MuZeroAgent
from cogment_verse_torch_agents.muzero.adapter import (
    MuZeroAgentAdapter,
    MuZeroSample,
    DEFAULT_MUZERO_RUN_CONFIG,
    make_workers,
    make_trial_configs,
)

from data_pb2 import EnvironmentSpecs

# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name
# pylint: disable=protected-access


@pytest.fixture
def lander_specs():
    return EnvironmentSpecs(implementation="gym/LunarLander-v2", num_players=1, num_input=8, num_action=4)


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_create(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    agent = MuZeroAgent(obs_dim=8, act_dim=4, device=device, run_config=DEFAULT_MUZERO_RUN_CONFIG)
    assert isinstance(agent.muzero, MuZero)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_act(device, env):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    agent = MuZeroAgent(obs_dim=8, act_dim=4, device=device, run_config=DEFAULT_MUZERO_RUN_CONFIG)
    model = agent.muzero
    model.eval()
    state = env.reset()

    for _ in range(100):
        observation = torch.from_numpy(state).to(device).float()
        action, _policy, _Q, _value = model.act(observation, 0.1, 0.3, 0.75, 0.995, 4, 32, 1.5, 15000.0)
        next_state, _reward, done, _info = env.step(action)
        if done:
            state = env.reset()
        else:
            state = next_state


# todo(jonathan): fix after refactor
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_learn(device, lander_specs):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    obs_dim = 8
    act_dim = 4

    agent_adapter = MuZeroAgentAdapter()
    agent, _agent_user_data = agent_adapter._create(
        "dummy_id",
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        run_config=DEFAULT_MUZERO_RUN_CONFIG,
        environment_specs=lander_specs,
    )

    model = agent.muzero
    optimizer = torch.optim.Adam(model.parameters())
    observation = torch.rand((4, 3, obs_dim))
    next_observation = torch.rand((4, 3, obs_dim))
    reward = torch.rand((4, 3))
    target_policy = torch.rand((4, 3, act_dim))
    target_value = torch.rand((4, 3))
    done = torch.rand((4, 3))
    action = torch.randint(low=0, high=act_dim, size=(4, 3))

    target_reward_probs = agent.muzero.reward_distribution.compute_target(reward).to(device)
    target_value_probs = agent.muzero.value_distribution.compute_target(target_value).to(device)

    _info = model.train_step(
        optimizer,
        observation,
        action,
        target_reward_probs,
        reward,
        next_observation,
        done,
        target_policy,
        target_value_probs,
        target_value,
        max_norm=1.0,
        s_weight=1.0,
        v_weight=1.0,
        _discount_factor=None,
        target_muzero=agent.target_muzero,
    )


def test_distributional():
    dist = Distributional(-2.0, 3.0, 8, 11)
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


def test_reward_transform():
    for val in [1.738, -1.738, 3.0, -2.0, -30, 40, -100, 200, -300, 500]:
        val = torch.tensor(val, dtype=torch.float32)
        transformed = reward_transform(val)
        val_2 = reward_transform_inverse(transformed)
        assert torch.allclose(val, val_2)


def test_distributional_transform():
    dist = Distributional(-100.0, 200.0, 4, 128, transform=reward_transform, inverse_transform=reward_transform_inverse)

    val = torch.tensor([1.738, -1.738, 3.0, -2.0, -30, 40, -100, 200]).to(torch.float32)
    val_transform = reward_transform(val)
    assert val_transform.shape == (8,)

    probs = dist.compute_target(val)
    assert probs.shape == (8, 128)
    probs_sum = torch.sum(probs, dim=1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum))

    val_transform_2 = torch.sum(probs * dist._bins.view(1, 128), dim=-1)
    assert val_transform_2.shape == (8,)

    val_2 = dist.compute_value(probs)
    assert val_2.shape == (8,)

    val_3 = reward_transform_inverse(val_transform_2)
    val_4 = reward_transform_inverse(val_transform)

    assert torch.allclose(val_transform, val_transform_2)
    assert torch.allclose(val, val_2)
    assert torch.allclose(val, val_3)
    assert torch.allclose(val, val_4)


def test_agentadapter(lander_specs):
    agent_adapter = MuZeroAgentAdapter()
    agent, _agent_user_data = agent_adapter._create(
        "dummy_id",
        obs_dim=8,
        act_dim=4,
        device="cpu",
        run_config=DEFAULT_MUZERO_RUN_CONFIG,
        environment_specs=lander_specs,
    )
    agent.act(torch.rand(8))

@pytest.mark.skip(reason="hangs on mac, doesn't wrk with torch on linux")
def test_workers(lander_specs):
    manager = mp.get_context("spawn")

    config = copy.deepcopy(DEFAULT_MUZERO_RUN_CONFIG)
    config.representation_network.hidden_size = 4
    config.dynamics_network.hidden_size = 4
    config.policy_network.hidden_size = 4
    config.value_network.hidden_size = 4
    config.projector_network.input_size = 4
    config.projector_network.hidden_size = 4
    config.training.batch_size = 4
    config.training.min_replay_buffer_size = 10
    config.mcts.max_depth = 2
    config.mcts.num_samples = 2
    agent_adapter = MuZeroAgentAdapter()
    agent, _agent_user_data = agent_adapter._create(
        "dummy_id",
        obs_dim=8,
        act_dim=4,
        device="cpu",
        run_config=config,
        environment_specs=lander_specs,
    )

    train_worker, replay_buffer, reanalyze_workers = make_workers(manager, agent, "muzero_test_model", config)
    workers = [train_worker, replay_buffer] + reanalyze_workers

    for worker in reanalyze_workers:
        worker.update_agent(agent)

    try:
        for worker in workers:
            worker.start()

        for trial_id in range(2):
            for step in range(config.training.min_replay_buffer_size + 1):
                state = np.random.rand(8).astype(np.float32)  # .reshape(1, -1)
                action = np.random.randint(4)
                reward = np.float32(np.random.rand())
                done = step == config.training.min_replay_buffer_size
                next_state = np.random.rand(8).astype(np.float32)  # .reshape(1, -1)
                policy = np.random.rand(4).astype(np.float32).reshape(1, -1)
                policy /= policy.sum()
                value = np.float32(np.random.rand())
                sample = MuZeroSample(state, action, reward, next_state, done, policy, value)
                replay_buffer.add_sample(f"{trial_id+1}", sample)

        time.sleep(5) # A little bit flaky, this is enough time so that the replay buffer worker has consumed the samples from 2 lines above
        assert replay_buffer.size() > config.training.min_replay_buffer_size

        for worker in workers:
            assert worker.is_alive()

        _info, _serialized_model = train_worker.results_queue.get(timeout=5.0)

        for worker in workers:
            assert worker.is_alive()
    finally:
        for worker in workers:
            worker.set_done(True)
        for worker in workers:
            worker.join(timeout=2.0)


def test_trial_configs():
    config = copy.deepcopy(DEFAULT_MUZERO_RUN_CONFIG)
    config.demonstration_trials = 10
    config.trial_count = 100
    trial_configs = make_trial_configs("mock_run_id", config, "mock_model_id", -1)
    assert len(trial_configs) == config.trial_count
