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

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List, Tuple, Union

import cogment
import numpy as np
import torch
from gym.spaces import Box, utils
from torch.distributions.normal import Normal

from cogment_verse import Model
from cogment_verse.run.run_session import RunSession
from cogment_verse.run.sample_producer_worker import SampleProducerSession
from cogment_verse.specs import PLAYER_ACTOR_CLASS, AgentConfig, EnvironmentConfig, EnvironmentSpecs, cog_settings

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


class ValueNetwork(torch.nn.Module):
    """Value network that quantifies the quality of an action given a state."""

    def __init__(self, num_input: int, num_hidden: int):
        super().__init__()
        self.input = torch.nn.Linear(num_input, num_hidden)
        self.fully_connected = torch.nn.Linear(num_hidden, num_hidden)
        self.output = torch.nn.Linear(num_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = self.input(x)
        x = torch.tanh(x)

        # Hidden layer
        x = self.fully_connected(x)
        x = torch.tanh(x)

        # Output layer
        value = self.output(x)

        return value


@dataclass
class NormalizationParams:
    """Normalization paramters for state and rewards"""

    mean_state: torch.Tensor  # Statistical mean for states
    var_state: torch.Tensor  # Statistical variance for states
    mean_reward: torch.Tensor  # Statistical mean for reward
    var_reward: torch.Tensor  # Statistical variance for reward


def initialize_weight(param) -> None:
    """Orthogonal initialization of the weight's values of a network"""

    if isinstance(param, torch.nn.Linear):
        torch.nn.init.orthogonal_(param.weight.data)
        torch.nn.init.constant_(param.bias.data, 0)


class Normalization:
    """Normalize the states and rewards on the fly
    Calulates the running mean and std of a data stream
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    Source: https://github.com/DLR-RM/stable-baselines3
    """

    def __init__(self, dtype: torch.FloatTensor = torch.float, epsilon: float = 1e-4, nums: int = 1):
        self.mean = torch.zeros(1, nums, dtype=dtype)
        self.var = torch.ones(1, nums, dtype=dtype)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, axis=0)
        batch_var = torch.var(x, axis=0)
        batch_count = x.size(dim=0)
        self.update_mean_var(batch_mean, batch_var, batch_count)

    def update_mean_var(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int):
        self.mean, self.var, self.count = self.update_mean_var_count(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @staticmethod
    def update_mean_var_count(
        mean: torch.Tensor,
        var: torch.Tensor,
        count: int,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ) -> Tuple[torch.Tensor, int]:

        """Update the statistical mean and variance"""
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        mean_a = var * count
        mean_b = batch_var * batch_count
        mean_2 = mean_a + mean_b + (delta**0.5) * count * batch_count / tot_count
        new_var = mean_2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class PPOModel(Model):
    """Proximal Policy Optimization (PPO) is an on-policy algorithm.
    https://arxiv.org/pdf/1707.06347.pdf.

    Attributes:
        model_id: Model identity
        environment_implementation: Environment type e.g., gym
        num_input: Number of states
        num_output: Number of actions
        policy_network_hidden_nodes: Number of hidden states for policy network
        value_network_hidden_nodes: Number of hidden states for value network
        learning_rate: Learning rate
        n_iter: Number of iterations
        dtype: Data type objects
        iteration: Version number of model
        policy_network: Policy network that outputs an action given a state
        value_network: Value network measure the the quality of action given a state
        policy_optimizer: Optimizer for policy network
        value_optimizer: Optimizer for value network
        policy_scheduler: Scheduling the learning rate for the policy network
        value_scheduler: Scheduling the learning rate for the value network
        state_normalization: Normalize state on the fly
        iter_idx: Attach each model to an index

    """

    state_normalization: Union[Normalization, None] = None

    def __init__(
        self,
        model_id: int,
        environment_implementation: str,
        num_input: int,
        num_output: int,
        policy_network_hidden_nodes: int = 64,
        value_network_hidden_nodes: int = 64,
        learning_rate: float = 0.01,
        n_iter: int = 1000,
        dtype=torch.float,
        iteration: int = 0,
        state_norm: bool = False,
    ) -> None:

        super().__init__(model_id, iteration)
        self._environment_implementation = environment_implementation
        self._num_input = num_input
        self._num_output = num_output
        self._policy_network_hidden_nodes = policy_network_hidden_nodes
        self._value_network_hidden_nodes = value_network_hidden_nodes
        self._dtype = dtype
        self._n_iter = n_iter
        self.state_norm = state_norm

        self.policy_network = PolicyNetwork(
            num_input=self._num_input,
            num_hidden=self._policy_network_hidden_nodes,
            num_output=self._num_output,
        ).to(self._dtype)

        self.value_network = ValueNetwork(
            num_input=num_input,
            num_hidden=self._value_network_hidden_nodes,
        ).to(self._dtype)

        # Intialize networks's parameters
        self.policy_network.apply(initialize_weight)
        self.value_network.apply(initialize_weight)

        # Get optimizer for two models
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=learning_rate,
        )

        # Learning schedule
        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.99)
        self.value_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.value_optimizer, gamma=0.99)

        # version user data
        self.iter_idx = 0
        self.total_samples = 0

    @property
    def state_norm(self) -> bool:
        return self._state_norm

    @state_norm.setter
    def state_norm(self, value: bool) -> None:
        self._state_norm = value
        if self._state_norm:
            self.state_normalization = Normalization(dtype=self._dtype, nums=self._num_input)

    def eval(self) -> None:
        self.policy_network.eval()
        self.value_network.eval()

    def get_model_user_data(self) -> dict:
        """Get user model"""
        return {
            "model_id": self.model_id,
            "environment_implementation": self._environment_implementation,
            "num_input": self._num_input,
            "num_output": self._num_output,
            "policy_network_hidden_nodes": self._policy_network_hidden_nodes,
            "value_network_hidden_nodes": self._value_network_hidden_nodes,
            "iter_idx": self.iter_idx,
            "total_samples": self.total_samples,
        }

    @staticmethod
    def serialize_model(model) -> bytes:
        stream = io.BytesIO()
        torch.save(
            (
                model.policy_network.state_dict(),
                model.value_network.state_dict(),
                model.get_model_user_data(),
            ),
            stream,
        )
        return stream.getvalue()

    @classmethod
    def deserialize_model(cls, serialized_model) -> PPOModel:
        stream = io.BytesIO(serialized_model)
        (policy_network_state_dict, value_network_state_dict, model_user_data) = torch.load(stream)

        model = cls(
            model_id=model_user_data["model_id"],
            environment_implementation=model_user_data["environment_implementation"],
            num_input=int(model_user_data["num_input"]),
            num_output=int(model_user_data["num_output"]),
            policy_network_hidden_nodes=int(model_user_data["policy_network_hidden_nodes"]),
            value_network_hidden_nodes=int(model_user_data["value_network_hidden_nodes"]),
        )
        model.policy_network.load_state_dict(policy_network_state_dict)
        model.value_network.load_state_dict(value_network_state_dict)
        model.iter_idx = model_user_data["iter_idx"]
        model.total_samples = model_user_data["total_samples"]

        return model


class PPOActor:
    """PPO actor"""

    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        """Get actor"""
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        # Start a session
        actor_session.start()

        config = actor_session.config

        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        observation_space = environment_specs.get_observation_space()
        action_space = environment_specs.get_action_space()

        assert isinstance(action_space.gym_space, Box)
        assert config.environment_specs.num_players == 1

        # Get model
        model = await PPOModel.retrieve_model(actor_session, config.model_id, config.model_iteration)
        model.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation = observation_space.deserialize(event.observation.observation)

                obs_tensor = torch.tensor(observation.flat_value, dtype=self._dtype).view(1, -1)

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
                    action_value = dist.sample().cpu().numpy()[0]

                # Send action to environment
                action = action_space.create(value=action_value)
                actor_session.do_action(action_space.serialize(action))


class PPOTraining:
    """Train PPO agent

    Atributes:
        _dtype: Data type object
        _environment_specs: Environment specs
        _cfg: Configuration i.e., hyper-parameters
        mse_loss: Loss for value network
        _device: Which device to send the data
        returns: Running rewards used for the normalization
        model: PPO model including the policy and value network
    """

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
        "grad_norm": 0.5,
        "state_norm": False,
    }

    def __init__(self, environment_specs: EnvironmentSpecs, cfg: EnvironmentConfig) -> None:
        super().__init__()
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg
        self._device = torch.device(self._cfg.device)
        self.returns = 0

        self.model = PPOModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            num_input=utils.flatdim(self._environment_specs.get_observation_space().gym_space),
            num_output=utils.flatdim(self._environment_specs.get_action_space().gym_space),
            learning_rate=self._cfg.learning_rate,
            n_iter=self._cfg.num_epochs,
            policy_network_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
            value_network_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
            dtype=self._dtype,
            state_norm=self._cfg.state_norm,
        )

    async def trial_sample_sequences_producer_impl(self, sample_producer_session: SampleProducerSession):
        """Collect sample from the trial"""

        # Share with A2C

        observation = []
        action = []
        reward = []
        done = []

        player_actor_params = sample_producer_session.trial_info.parameters.actors[0]

        player_actor_name = player_actor_params.name
        player_environment_specs = EnvironmentSpecs.deserialize(player_actor_params.config.environment_specs)
        player_observation_space = player_environment_specs.get_observation_space()
        player_action_space = player_environment_specs.get_action_space()

        async for sample in sample_producer_session.all_trial_samples():
            if sample.trial_state == cogment.TrialState.ENDED:
                # This sample includes the last observation and no action
                # The last sample was the last useful one
                done[-1] = torch.ones(1, dtype=self._dtype)
                break

            actor_sample = sample.actors_data[player_actor_name]
            observation.append(
                torch.tensor(player_observation_space.deserialize(actor_sample.observation).value, dtype=self._dtype)
            )

            action.append(torch.tensor(player_action_space.deserialize(actor_sample.action).value, dtype=self._dtype))
            reward.append(
                torch.tensor(actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype)
            )
            done.append(torch.zeros(1, dtype=self._dtype))

        # Keeping the samples grouped by trial by emitting only one grouped sample at the end of the trial
        sample_producer_session.produce_sample((observation, action, reward, done))

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish model the model"""

        model_id = f"{run_session.run_id}_model"

        assert self._environment_specs.num_players == 1
        assert isinstance(self._environment_specs.get_action_space().gym_space, Box)
        assert self._cfg.num_steps >= self._cfg.batch_size

        # Initalize model
        self.model.model_id = model_id
        serialized_model = PPOModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
            policy_network_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
            value_network_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
        )

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx: int, iter_idx: int):
            agent_actor_params = cogment.ActorParameters(
                cog_settings,
                name="player",
                class_name=PLAYER_ACTOR_CLASS,
                implementation="actors.ppo.PPOActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                    model_id=model_id,
                    model_iteration=iteration_info.iteration,
                ),
            )

            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id,
                    render=False,
                    seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
                ),
                actors=[agent_actor_params],
            )

        # Run environment
        observations = []
        actions = []
        rewards = []
        dones = []
        episode_rewards = []
        for iter_idx in range(self._cfg.num_iter):
            for (_, _, _, sample) in run_session.start_and_await_trials(
                trials_id_and_params=[
                    (f"{run_session.run_id}_{iter_idx}_{trial_idx}", create_trial_params(trial_idx, iter_idx))
                    for trial_idx in range(self._cfg.epoch_num_trials)
                ],
                sample_producer_impl=self.trial_sample_sequences_producer_impl,
                num_parallel_trials=self._cfg.num_parallel_trials,
            ):
                # Collect the rollout
                (trial_observation, trial_action, trial_reward, trial_done) = sample

                observations.extend(trial_observation)
                actions.extend(trial_action)
                rewards.extend(trial_reward)
                dones.extend(trial_done)
                episode_rewards.append(torch.vstack(trial_reward).sum())

                # Publish the newly trained version every 100 steps
                if len(actions) >= self._cfg.num_steps * self._cfg.epoch_num_trials + 1:
                    # Update model parameters
                    policy_loss, value_loss = await self.train_step(
                        observations=observations, rewards=rewards, actions=actions, dones=dones
                    )

                    # Reset the data storage
                    observations = []
                    actions = []
                    rewards = []
                    dones = []
                    if iter_idx % 100 == 0:
                        # Compute average rewards for last 100 episodes
                        avg_rewards = await self.compute_average_reward(episode_rewards)
                        log.info(
                            f"epoch #{iter_idx + 1}/{self._cfg.num_iter}: [policy loss: {policy_loss:0.2f}, value loss: {value_loss:0.2f}, avg. rewards: {avg_rewards:0.2f}]"
                        )

                        run_session.log_metrics(
                            model_iteration=iteration_info.iteration,
                            policy_loss=policy_loss.item(),
                            value_loss=value_loss.item(),
                            rewards=avg_rewards.item(),
                        )

                    # Publish the newly updated model
                    self.model.iter_idx = iter_idx
                    serialized_model = PPOModel.serialize_model(self.model)
                    iteration_info = await run_session.model_registry.store_model(
                        name=model_id,
                        model=serialized_model,
                    )

    async def train_step(
        self,
        observations: List[torch.Tensor],
        rewards: List[torch.Tensor],
        actions: List[torch.Tensor],
        dones: List[torch.Tensor],
    ):
        """Train the model after collecting the data from the trial"""

        # Take n steps from the rollout
        observations = torch.vstack(observations)[: self._cfg.num_steps * self._cfg.epoch_num_trials + 1]
        actions = torch.vstack(actions)[: self._cfg.num_steps * self._cfg.epoch_num_trials]
        rewards = torch.vstack(rewards)[: self._cfg.num_steps * self._cfg.epoch_num_trials]
        dones = torch.vstack(dones)[: self._cfg.num_steps * self._cfg.epoch_num_trials]

        # Normalize the observations
        if self.model.state_normalization is not None:
            self.model.state_normalization.update(observations)

        # Make a dataloader in order to process data in batch
        batch_state = self.make_dataloader(observations[:-1], self._cfg.batch_size, self.model._num_input)
        batch_action = self.make_dataloader(actions, self._cfg.batch_size, self.model._num_output)

        values = self.compute_value(batch_state)
        with torch.no_grad():
            next_value = self.model.value_network(observations[-1:])
        next_value = next_value * (1 - dones[-1])
        values = torch.cat((values, next_value), dim=0)

        log_probs = self.compute_log_lik(batch_state, batch_action)

        # Compute the generalized advantage estimation
        advs = self.compute_gae(
            rewards=rewards, values=values, dones=dones, gamma=self._cfg.discount_factor, lam=self._cfg.lambda_gae
        )

        # Update parameters for policy and value networks
        policy_loss, value_loss = self.update_parameters(
            states=observations[:-1],
            actions=actions,
            advs=advs,
            values=values[:-1],
            log_probs=log_probs,
            num_epochs=self._cfg.num_epochs,
        )

        return policy_loss, value_loss

    def update_parameters(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advs: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        num_epochs: int,
    ) -> Tuple[torch.Tensor]:
        """Update policy & value networks"""

        returns = advs + values
        num_states = len(returns)
        for _ in range(num_epochs):
            for _ in range(num_states // self._cfg.batch_size):
                # Get data in batch. TODO: Send data to device (need to test with cuda)
                idx = np.random.randint(0, num_states, self._cfg.batch_size)
                state = states[idx].to(self._device)
                action = actions[idx].to(self._device)
                return_ = returns[idx].to(self._device)
                adv = advs[idx].to(self._device)
                old_log_prob = log_probs[idx].to(self._device)

                # Compute the value and values loss
                value = self.model.value_network(state)
                value_loss = torch.nn.functional.mse_loss(return_, value) * self._cfg.value_loss_coef

                # Get action distribution & the log-likelihood
                action_dist, _ = self.model.policy_network(state)
                new_log_prob = action_dist.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_prob)

                # Compute policy loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * torch.clamp(ratio, 1 - self._cfg.clipping_coef, 1 + self._cfg.clipping_coef)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Update value network
                self.model.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.value_network.parameters(), self._cfg.grad_norm)
                self.model.value_optimizer.step()

                # Udapte policy network
                self.model.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.policy_network.parameters(), self._cfg.grad_norm)
                self.model.policy_optimizer.step()

        # Decaying learning rate after each update
        self.model.policy_scheduler.step()
        self.model.value_scheduler.step()

        return policy_loss, value_loss

    @staticmethod
    async def compute_average_reward(rewards: list) -> float:
        """Compute the average reward of the last 100 episode"""
        last_100_rewards = rewards[np.maximum(0, len(rewards) - 100) : len(rewards)]
        return torch.vstack(last_100_rewards).mean()

    def compute_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute values given the states"""
        values = []
        for obs in observations:
            with torch.no_grad():
                values.append(self.model.value_network(obs))

        return torch.vstack(values)

    def compute_log_lik(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood for each actions"""
        log_probs = []
        for obs, action in zip(observations, actions):
            with torch.no_grad():
                dist, _ = self.model.policy_network(obs)
            log_probs.append(dist.log_prob(action))
        return torch.vstack(log_probs)

    @staticmethod
    async def get_n_steps_data(dataset: torch.Tensor, num_steps: int) -> torch.tensor:
        """Get the data up to nth steps"""
        return dataset[:num_steps]

    def make_dataloader(self, dataset: torch.Tensor, batch_size: int, num_obs: int) -> List[torch.Tensor]:
        """Create a dataloader in batches"""
        # Initialization
        output_batches = torch.zeros((batch_size, num_obs), dtype=self._dtype)
        num_data = len(dataset)
        data_loader = []
        count = 0
        for i, y_batch in enumerate(dataset):
            output_batches[count, :] = y_batch
            # Store data
            if (i + 1) % batch_size == 0:
                data_loader.append(output_batches)

                # Reset
                count = 0
                output_batches = torch.zeros((batch_size, num_obs), dtype=self._dtype)
            else:
                count += 1
                if i == num_data - 1:
                    data_loader.append(output_batches[:count, :])

        return data_loader

    def normalize_rewards(self, rewards: list, model: PPOModel) -> list:
        """Normalize the rewards"""
        normalized_reward = []
        for rew in rewards:
            normalized_reward.append(rew / (model.reward_normalizaiton.var + 1e-8) ** 0.5)
            self.returns = self.returns * self._cfg.discount_factor + rew
            model.reward_normalization.update(self.returns)

        return normalized_reward

    @staticmethod
    def compute_gae(
        rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float = 0.99, lam: float = 0.95
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation. See equations 11 & 12 in
        https://arxiv.org/pdf/1707.06347.pdf
        """

        advs = []
        gae = 0.0
        dones = torch.cat((dones, torch.zeros(1, 1)), dim=0)
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * (values[i + 1]) * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advs.append(gae)
        advs.reverse()
        return torch.vstack(advs)
