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

    def get_model_user_data(self) -> dict:
        """Get user model"""
        return {
            "environment_implementation": self.environment_implementation,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "policy_network_hidden_nodes": self.policy_network_hidden_nodes,
            "value_network_hidden_nodes": self.value_network_hidden_nodes,
        }

    def save(self, model_data_f: str) -> dict:
        """Save the model"""
        torch.save((self.policy_network.state_dict(), self.value_network.state_dict()), model_data_f)
        return {"iter_idx": self.iter_idx, "total_samples": self.total_samples}

    @classmethod
    def load(
        cls, model_id: int, version_number: int, model_user_data: dict, version_user_data: dict, model_data_f: str
    ) -> Model:
        """Load the model"""
        model = SACModel(
            model_id=model_id,
            version_number=version_number,
            environment_implementation=model_user_data["environment_implementation"],
            num_inputs=int(model_user_data["num_inputs"]),
            num_outputs=int(model_user_data["num_outputs"]),
            policy_network_hidden_nodes=int(model_user_data["policy_network_hidden_nodes"]),
            value_network_hidden_nodes=int(model_user_data["value_network_hidden_nodes"]),
        )

        # Load the model parameters
        (policy_network_state_dict, value_network_state_dict) = torch.load(model_data_f)
        model.policy_network.load_state_dict(policy_network_state_dict)
        model.value_network.load_state_dict(value_network_state_dict)

        # Load version data
        model.iter_idx = version_user_data["iter_idx"]
        model.total_samples = version_user_data["total_samples"]
        return model


class SACActor:
    """Soft actor critic actor"""

    def __init__(self):
        self._dtype = torch.float

    def get_actor_classes(self):
        """Get actor"""
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        # Start a session
        actor_session.start()
        config = actor_session.config
        assert config.environment_specs.num_players == 1
        assert len(config.environment_specs.action_space.properties) == 1
        assert config.environment_specs.action_space.properties[0].WhichOneof("type") == "box"

        # Get observation and action space
        observation_space = config.environment_specs.observation_space
        action_space = config.environment_specs.action_space

        # Get model
        model, _, _ = await actor_session.model_registry.retrieve_version(
            SACModel, config.model_id, config.model_version
        )

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                obs_tensor = torch.tensor(
                    flatten(observation_space, event.observation.observation.value), dtype=self._dtype
                ).view(1, -1)

                # Normalize the observation
                if model.state_normalization is not None:
                    obs_tensor = torch.clamp(
                        (obs_tensor - model.state_normalization.mean) / (model.state_normalization.var + 1e-8) ** 0.5,
                        min=-10,
                        max=10,
                    )

                # Get action from policy network
                with torch.no_grad():
                    dist, _ = model.policy_network(obs_tensor)
                    action = dist.sample().cpu().numpy()[0]

                # Send action to environment
                action_value = unflatten(action_space, action)
                actor_session.do_action(PlayerAction(value=action_value))


class SACTraining:
    """Train SAC agent"""

    default_cfg = {
        "seed": 10,
        "num_epochs": 10,
        "num_iter": 500,
        "epoch_num_trials": 1,
        "num_parallel_trials": 1,
        "discount_factor": 0.99,
        "entropy_loss_coef": 0.05,
        "value_loss_coef": 0.5,
        "action_loss_coef": 1.0,
        "clipping_coef": 0.1,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "num_steps": 2048,
        "lambda_gae": 0.95,
        "device": "cpu",
        "policy_network": {"num_hidden_nodes": 64},
        "value_network": {"num_hidden_nodes": 64},
    }

    def __init__(self, environment_specs: EnvironmentSpecs, cfg: EnvironmentConfig) -> None:
        super().__init__()
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg
        self._device = torch.device(self._cfg.device)
        self.returns = 0

        self.model = SACModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            num_inputs=flattened_dimensions(self._environment_specs.observation_space),
            num_outputs=flattened_dimensions(self._environment_specs.action_space),
            learning_rate=self._cfg.learning_rate,
            policy_network_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
            value_network_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
            dtype=self._dtype,
        )

    async def trial_sample_sequences_producer_impl(self, sample_producer_session: SampleProducerSession):
        """Collect sample from the trial"""
        observation = []
        action = []
        reward = []
        done = []

        player_actor_params = sample_producer_session.trial_info.parameters.actors[0]
        player_actor_name = player_actor_params.name
        player_observation_space = player_actor_params.config.environment_specs.observation_space
        player_action_space = player_actor_params.config.environment_specs.action_space

        async for sample in sample_producer_session.all_trial_samples():
            if sample.trial_state == cogment.TrialState.ENDED:
                # This sample includes the last observation and no action
                # The last sample was the last useful one
                done[-1] = torch.ones(1, dtype=self._dtype)
                break

            # TODO: collect data in buffer
