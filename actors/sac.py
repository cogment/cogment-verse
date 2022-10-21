# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import logging
from dataclasses import dataclass
from typing import List, Tuple, Union

import cogment
import numpy as np
import torch
from torch.distributions.normal import Normal

from cogment_verse import Model
from cogment_verse.run.run_session import RunSession
from cogment_verse.run.sample_producer_worker import SampleProducerSession
from cogment_verse.specs import (
    PLAYER_ACTOR_CLASS,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentSpecs,
    PlayerAction,
    cog_settings,
    flatten,
    flattened_dimensions,
    unflatten,
)

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)

# pylint: disable=E1102
# pylint: disable=W0212
class PolicyNetwork(torch.nn.Module):
    """Gaussian policy network"""

    def __init__(self, num_input: int, num_output: int, num_hidden: int) -> None:
        super().__init__()
        self.input = torch.nn.Linear(num_input, num_hidden)
        self.fully_connected = torch.nn.Linear(num_hidden, num_hidden)
        self.mean = torch.nn.Linear(num_hidden, num_output)
        self.log_std = torch.nn.Parameter(torch.zeros(1, num_output))

    def forward(self, x: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        # Input layer
        x = self.input(x)
        x = torch.tanh(x)

        # Hidden layer
        x = self.fully_connected(x)
        x = torch.tanh(x)

        # Output layer
        mean = self.mean(x)
        std = self.log_std.exp()
        dist = torch.distributions.normal.Normal(mean, std)

        return dist, mean


class DeterministicPolicyNetwork:
    """ "Deterministic policy network"""

    def __init__(self, num_input: int, num_output: int, num_hidden: int) -> None:
        super().__init__()
        self.input = torch.nn.Linear(num_input, num_hidden)
        self.fully_connected = torch.nn.Linear(num_hidden, num_hidden)
        self.output = torch.nn.Linear(num_hidden, num_output)

    def forward(self, x: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        # Input layer
        x = self.input(x)
        x = torch.tanh(x)

        # Hidden layer
        x = self.fully_connected(x)
        x = torch.tanh(x)

        # Output layer
        action = self.output(x)

        return action


class ValueNetwork(torch.nn.Module):
    """Value network that quantifies the quality of an action given a state."""

    def __init__(self, num_input: int, num_hidden: int):
        super().__init__()
        self.input = torch.nn.Linear(num_input, num_hidden)
        self.fully_connected = torch.nn.Linear(num_hidden, num_hidden)
        self.output = torch.nn.Linear(num_hidden, 1)

    def forward(self, x: torch.Tensor):
        # Input layer
        x = self.input(x)
        x = torch.tanh(x)

        # Hidden layer
        x = self.fully_connected(x)
        x = torch.tanh(x)

        # Output layer
        value = self.output(x)

        return value


def initialize_weight(param) -> None:
    """Orthogonal initialization of the weight's values of a network"""

    if isinstance(param, torch.nn.Linear):
        torch.nn.init.orthogonal_(param.weight.data)
        torch.nn.init.constant_(param.bias.data, 0)


class SACModel(Model):
    """Soft-Actor Critic (SAC) https://arxiv.org/abs/1801.01290"""

    def __init__(
        self,
        model_id: int,
        environment_implementation: str,
        num_inputs: int,
        num_outputs: int,
        policy_network_hidden_nodes: int,
        value_network_hidden_nodes: int,
        learning_rate: float = 0.01,
        dtype=torch.float,
        version_number: int = 0,
    ) -> None:
        self.model_id = model_id
        self.environment_implementation = environment_implementation
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.policy_network_hidden_nodes = policy_network_hidden_nodes
        self.value_network_hidden_nodes = value_network_hidden_nodes
        self.learning_rate = learning_rate
        self.dtype = dtype
        self.version_number = version_number

        self.policy_network = PolicyNetwork(
            num_input=self.num_inputs,
            num_hidden=self.policy_network_hidden_nodes,
            num_output=self.num_outputs,
        ).to(self.dtype)

        self.value_network = ValueNetwork(
            num_input=num_inputs,
            num_hidden=self.value_network_hidden_nodes,
        ).to(self.dtype)

        # Intialize networks's parameters
        self.policy_network.apply(initialize_weight)
        self.value_network.apply(initialize_weight)

        # Get optimizer for two models
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=learning_rate)

        # Learning schedule
        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.99)
        self.value_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.value_optimizer, gamma=0.99)

        # version user data
        self.iter_idx = 0
        self.total_samples = 0
