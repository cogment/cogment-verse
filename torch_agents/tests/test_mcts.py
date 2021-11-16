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

from lib.mcts import MCTS

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


def perceptron(num_in, num_hidden, num_out):
    return torch.nn.Sequential(
        torch.nn.Linear(num_in, num_hidden), torch.nn.ReLU(), torch.nn.Linear(num_hidden, num_out)
    )


@pytest.fixture
def num_latent():
    return 8


@pytest.fixture
def num_action():
    return 4


@pytest.fixture
def policy(num_latent, num_action):
    return torch.nn.Sequential(perceptron(num_latent, 16, num_action), torch.nn.Softmax(dim=1))


@pytest.fixture
def dynamics(num_latent, num_action):
    class Dynamics(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._perceptron = perceptron(num_latent + num_action, 16, num_latent + 1)

        def forward(self, representation, action):
            action = torch.nn.functional.one_hot(action, num_action)
            x = torch.cat((representation, action), dim=1)
            return torch.split(self._perceptron(x), [num_latent, 1], dim=1)

    return Dynamics()


@pytest.fixture
def value(num_latent):
    return perceptron(num_latent, 16, 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mcts(policy, dynamics, value, num_latent, num_action, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip()

    device = torch.device(device)
    dynamics = dynamics.to(device)
    policy = policy.to(device)
    value = value.to(device)

    representation = torch.rand(1, num_latent).to(device)
    root = MCTS(policy=policy, value=value, dynamics=dynamics, representation=representation, max_depth=8)
    root.build_search_tree(100)
    policy, value = root.improved_targets()
    assert policy.shape == (1, num_action)
    assert value.shape == ()
