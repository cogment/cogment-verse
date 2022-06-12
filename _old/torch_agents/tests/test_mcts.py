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
import torch

from cogment_verse_torch_agents.muzero.mcts import MCTS

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name
# pylint: disable=no-self-use


@pytest.fixture
def mock_policy():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_actions = 2

        def forward(self, x):
            # uniform + small random perturbation
            prob = torch.ones((x.shape[0], self.num_actions), dtype=torch.float32)
            prob += 0.01 * torch.rand_like(prob)
            prob /= self.num_actions
            return prob

    return Policy()


@pytest.fixture
def mock_dynamics():
    # XOR game to test correctness of MCTS procedure:
    # - two states 0,1 and two actions 0,1
    # - reward is equal to state xor action
    # - next_state is equal to action
    class Dynamics(torch.nn.Module):
        def forward(self, state, action):
            action = torch.nn.functional.one_hot(action, 2)
            reward = torch.sum((1 - state) * action, dim=1)
            next_state = action
            return next_state, reward

    return Dynamics()


@pytest.fixture
def mock_value():
    class Value(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0])

    return Value()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mcts(device, mock_policy, mock_value, mock_dynamics):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    device = torch.device(device)
    mock_dynamics = mock_dynamics.to(device)
    mock_policy = mock_policy.to(device)
    mock_value = mock_value.to(device)

    for state in [[0, 1], [1, 0]]:
        representation = torch.tensor(state).unsqueeze(0)
        root = MCTS(
            policy=mock_policy, value=mock_value, dynamics=mock_dynamics, representation=representation, max_depth=8
        )
        root.build_search_tree(100)
        policy, _q, value = root.improved_targets(0.75)
        assert policy.shape == (1, 2)
        assert value.shape == ()
        action = torch.argmax(policy).cpu().item()
        wrong_action = torch.argmax(torch.tensor(state)).item()
        assert action != wrong_action
