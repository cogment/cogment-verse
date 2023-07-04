# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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
from typing import Tuple, Union

import cogment
import numpy as np
import torch
from cogment.actor import ActorSession
from gym.spaces import utils
from omegaconf import DictConfig, ListConfig
from torch.distributions.normal import Normal

from cogment_verse import Model, PPOReplayBuffer
from cogment_verse.run.run_session import RunSession
from cogment_verse.run.sample_producer_worker import SampleProducerSession
from cogment_verse.specs import PLAYER_ACTOR_CLASS, AgentConfig, EnvironmentConfig, EnvironmentSpecs, cog_settings

torch.multiprocessing.set_sharing_strategy("file_system")


log = logging.getLogger(__name__)
LOG_STD_MAX = 2
LOG_STD_MIN = -20


# pylint: disable=protected-access
# pylint: disable=abstract-method
# pylint: disable=unused-argument
# pylint: disable=unused-variable
# pylint: disable=too-many-lines
# pylint: disable=not-callable
def initialize_layer(layer: torch.nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_device(device: str) -> str:
    """Device setup"""
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"

    return "cpu"


class PolicyNetwork(torch.nn.Module):
    """Gaussian policy network"""

    def __init__(self, num_input: int, num_output: int, num_hidden: int) -> None:
        super().__init__()
        self.input = initialize_layer(torch.nn.Linear(num_input, num_hidden))
        self.fully_connected = initialize_layer(torch.nn.Linear(num_hidden, num_hidden))
        self.mean = initialize_layer(torch.nn.Linear(num_hidden, num_output), std=0.01)
        self.log_std = torch.nn.Parameter(torch.ones(1, num_output) * (-0.5))

    def forward(self, x: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        # Input layer
        x = self.input(x)
        x = torch.tanh(x)

        # Hidden layer
        x = self.fully_connected(x)
        x = torch.tanh(x)

        # Output layer
        mean = self.mean(x)
        log_std_val = torch.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = log_std_val.exp()
        dist = torch.distributions.normal.Normal(mean, std)

        return dist, mean


class ValueNetwork(torch.nn.Module):
    """Value network that quantifies the quality of an action given a state."""

    def __init__(self, num_input: int, num_hidden: int):
        super().__init__()
        self.input = initialize_layer(torch.nn.Linear(num_input, num_hidden))
        self.fully_connected = initialize_layer(torch.nn.Linear(num_hidden, num_hidden))
        self.output = initialize_layer(torch.nn.Linear(num_hidden, 1), std=1)

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


class Normalization:
    """Normalize the states and rewards on the fly
    Calculate the running mean and std of a data stream
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    Source: https://github.com/DLR-RM/stable-baselines3
    """

    max_val = 10
    min_val = -10

    def __init__(
        self,
        dtype: torch.FloatTensor = torch.float,
        epsilon: float = 1e-4,
        nums: int = 1,
        mean: Union[torch.Tensor, None] = None,
        var: Union[torch.Tensor, None] = None,
    ):
        self._nums = nums
        self._dtype = dtype
        self.mean = mean
        self.var = var
        self.count = epsilon

    @property
    def mean(self) -> Union[torch.Tensor, None]:
        """Get running mean"""
        return self._mean

    @mean.setter
    def mean(self, value: Union[torch.Tensor, None]) -> None:
        """Set running mean"""
        if value is not None:
            self._mean = value
        else:
            self._mean = torch.zeros(1, self._nums, dtype=self._dtype)

    @property
    def var(self) -> Union[torch.Tensor, None]:
        """Get running variance"""
        return self._var

    @var.setter
    def var(self, value: Union[torch.Tensor, None]) -> None:
        """Set running mean"""
        if value is not None:
            self._var = value
        else:
            self._var = torch.ones(1, self._nums, dtype=self._dtype)

    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, axis=0)
        batch_var = torch.var(x, axis=0)
        batch_count = x.size(dim=0)
        self.update_mean_var(batch_mean, batch_var, batch_count)

    def update_mean_var(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int):
        self.mean, self.var, self.count = self.update_mean_var_count(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize the state"""
        return torch.clamp(
            (observation.clone() - self.mean) / (self.var + 1e-8) ** 0.5, min=self.min_val, max=self.max_val
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


class APPOModel(Model):
    """Asynchronous PPO"""

    state_normalizer: Union[Normalization, None] = None

    def __init__(
        self,
        model_id: int,
        environment_implementation: str,
        num_actions: int,
        num_inputs: tuple,
        hidden_nodes: int = 64,
        n_iter: int = 1000,
        dtype=torch.float32,
        device: str = "cpu",
        iteration: int = 0,
        state_norm: bool = False,
    ) -> None:
        super().__init__(model_id, iteration)
        self._environment_implementation = environment_implementation
        self._num_actions = num_actions
        self.num_inputs = num_inputs
        self._hidden_nodes = hidden_nodes
        self.device = device
        self._dtype = dtype
        self._n_iter = n_iter
        self.state_norm = state_norm

        self.policy_network = PolicyNetwork(
            num_input=self.num_inputs, num_hidden=self._hidden_nodes, num_output=self._num_actions
        ).to(self._dtype)
        self.value_network = ValueNetwork(num_input=self.num_inputs, num_hidden=self._hidden_nodes).to(self._dtype)

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
            self.state_normalizer = Normalization(dtype=self._dtype, nums=self.num_inputs)

    def eval(self) -> None:
        self.policy_network.eval()
        self.value_network.eval()

    def get_model_user_data(self) -> dict:
        """Get user model"""
        return {
            "model_id": self.model_id,
            "environment_implementation": self._environment_implementation,
            "num_actions": self._num_actions,
            "num_inputs": self.num_inputs,
            "hidden_nodes": self._hidden_nodes,
            "device": self.device,
            "iter_idx": self.iter_idx,
            "total_samples": self.total_samples,
            "running_mean": self.state_normalizer.mean.clone().numpy(),
            "running_var": self.state_normalizer.var.clone().numpy(),
            "state_norm": self.state_norm,
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
    def deserialize_model(cls, serialized_model) -> APPOModel:
        stream = io.BytesIO(serialized_model)
        (policy_network_state_dict, value_network_state_dict, model_user_data) = torch.load(stream)

        model = APPOModel(
            model_id=model_user_data["model_id"],
            environment_implementation=model_user_data["environment_implementation"],
            num_actions=model_user_data["num_actions"],
            num_inputs=model_user_data["num_inputs"],
            hidden_nodes=int(model_user_data["hidden_nodes"]),
            device="cpu",
            state_norm=bool(model_user_data["state_norm"]),
        )
        if model.state_normalizer is not None:
            model.state_normalizer.mean = torch.tensor(model_user_data["running_mean"])
            model.state_normalizer.var = torch.tensor(model_user_data["running_var"])

        model.policy_network.load_state_dict(policy_network_state_dict)
        model.value_network.load_state_dict(value_network_state_dict)
        model.iter_idx = model_user_data["iter_idx"]
        model.total_samples = model_user_data["total_samples"]

        return model


class APPOActor:
    """PPO actor"""

    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        """Get actor"""
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session: ActorSession):
        # Start a session
        actor_session.start()
        config = actor_session.config

        # Setup random seed
        torch.manual_seed(config.seed)

        # Get observation and action space
        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        observation_space = environment_specs.get_observation_space()
        action_space = environment_specs.get_action_space(seed=config.seed)

        # Get model
        model = await APPOModel.retrieve_model(actor_session.model_registry, config.model_id, config.model_iteration)
        model.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                # Retrieve the model every N rollout steps
                if (
                    config.model_update_frequency != 0
                    and (event.observation.tick_id + 1) % config.model_update_frequency == 0
                ):
                    model = await APPOModel.retrieve_model(actor_session.model_registry, config.model_id, -1)
                    model.eval()

                observation = observation_space.deserialize(event.observation.observation)
                obs_tensor = torch.tensor(observation.flat_value, dtype=self._dtype).view(1, -1)

                # Normalize the observation
                if model.state_normalizer is not None:
                    obs_tensor = model.state_normalizer.normalize(obs_tensor)

                # Get action from policy network
                with torch.no_grad():
                    dist, _ = model.policy_network(obs_tensor)
                    action_value = dist.sample().cpu().numpy()[0]

                # Send action to environment
                action = action_space.create(value=action_value)
                actor_session.do_action(action_space.serialize(action))


class RolloutBuffer:
    """Rollout buffer for PPO"""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple,
        action_shape: tuple,
        observation_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.observation_dtype = observation_dtype
        self.action_dtype = action_dtype
        self.reward_dtype = reward_dtype

        self.observations = torch.zeros((self.capacity, *self.observation_shape), dtype=self.observation_dtype)
        self.actions = torch.zeros((self.capacity, *self.action_shape), dtype=self.action_dtype)
        self.rewards = torch.zeros((self.capacity,), dtype=self.reward_dtype)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32)

        self._ptr = 0
        self.num_total = 0

    def add(self, observation: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None:
        """Add samples to rollout buffer"""
        if self.num_total < self.capacity:
            self.observations[self._ptr] = observation
            self.actions[self._ptr] = action
            self.rewards[self._ptr] = reward
            self.dones[self._ptr] = done
            self._ptr = (self._ptr + 1) % self.capacity
            self.num_total += 1

    def reset(self) -> None:
        """Reset the rollout"""
        self.observations = torch.zeros((self.capacity, *self.observation_shape), dtype=self.observation_dtype)
        self.actions = torch.zeros((self.capacity, *self.action_shape), dtype=self.action_dtype)
        self.rewards = torch.zeros((self.capacity,), dtype=self.reward_dtype)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32)
        self._ptr = 0
        self.num_total = 0


class APPOTraining:
    """Train APPO agent"""

    default_cfg = {
        "seed": 3407,
        "num_epochs": 10,
        "num_iter": 500,
        "epoch_num_trials": 1,
        "num_parallel_trials": 1,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "buffer_size": 10000,
        "learning_starts": 1,
        "update_freq": 1,
        "num_rollout_steps": 2048,
        "max_training_steps": 300_000,
        "discount_factor": 0.99,
        "lambda_gae": 0.95,
        "device": "cpu",
        "entropy_loss_coef": 0.05,
        "value_loss_coef": 0.5,
        "clipping_coef": 0.1,
        "num_hidden_nodes": 64,
        "grad_norm": 0.5,
        "logging_interval": 100,
        "state_norm": False,
    }

    def __init__(self, environment_specs: EnvironmentSpecs, cfg: Union[ListConfig, DictConfig]) -> None:
        super().__init__()
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg
        available_device = get_device(self._cfg.device)
        self._torch_device = torch.device(available_device)
        self.returns = 0
        self.model_id = ""

        torch.manual_seed(cfg.seed)
        self.model = APPOModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            num_actions=utils.flatdim(self._environment_specs.get_action_space().gym_space),
            num_inputs=utils.flatdim(self._environment_specs.get_observation_space().gym_space),
            hidden_nodes=self._cfg.num_hidden_nodes,
            n_iter=self._cfg.num_epochs,
            dtype=self._dtype,
            state_norm=self._cfg.state_norm,
        )

        # Get optimizer for two models
        self.policy_optimizer = torch.optim.Adam(self.model.policy_network.parameters(), lr=self._cfg.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.model.value_network.parameters(), lr=self._cfg.learning_rate)

    @staticmethod
    def should_load_model(step: int, num_rollout_steps: int, trial_done: bool) -> bool:
        """Load model from model registry if it matches predefined conditions regarding
        trial status and tick id"""
        is_multiple_of_rollout = (step + 1) % num_rollout_steps == 0
        is_first_step = step == 0
        is_loaded = is_multiple_of_rollout and not trial_done
        return is_loaded or is_first_step

    async def sample_producer_impl(self, sample_producer_session: SampleProducerSession):
        """Collect sample from the trial"""
        actor_params = {
            actor_params.name: actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == PLAYER_ACTOR_CLASS
        }
        actor_names = list(actor_params.keys())
        player_environment_specs = EnvironmentSpecs.deserialize(actor_params[actor_names[0]].config.environment_specs)
        player_observation_space = player_environment_specs.get_observation_space()
        player_action_space = player_environment_specs.get_action_space()

        # Load the model
        model = await APPOModel.retrieve_model(sample_producer_session.model_registry, self.model_id, -1)
        model.eval()

        rollout_buffer = RolloutBuffer(
            capacity=self._cfg.num_rollout_steps,
            observation_shape=(utils.flatdim(player_observation_space.gym_space),),
            action_shape=(utils.flatdim(player_action_space.gym_space),),
            action_dtype=torch.float32,
        )

        values = []
        log_probs = []
        current_players = []
        episode_rewards = []
        done = torch.zeros(1, dtype=torch.int8)
        player_actor_name = actor_names[0]
        total_reward = 0
        step = 0
        async for sample in sample_producer_session.all_trial_samples():
            # Trail status
            trial_done = sample.trial_state == cogment.TrialState.ENDED
            actor_sample = sample.actors_data[player_actor_name]

            # Load model
            if self.should_load_model(sample.tick_id, self._cfg.num_rollout_steps, trial_done):
                model = await APPOModel.retrieve_model(sample_producer_session.model_registry, self.model_id, -1)
                model.eval()

            # Collect data from environment
            obs = torch.tensor(player_observation_space.deserialize(actor_sample.observation).value, dtype=self._dtype)
            done = torch.ones(1, dtype=torch.float32) if trial_done else torch.zeros(1, dtype=torch.float32)
            reward = (
                torch.tensor(actor_sample.reward, dtype=self._dtype)
                if actor_sample.reward is not None
                else torch.tensor(0, dtype=self._dtype)
            )

            if not trial_done:
                action = torch.tensor(player_action_space.deserialize(actor_sample.action).value, dtype=self._dtype)
                current_players.append(player_actor_name)

            # Compute values and log probs
            if step % self._cfg.num_rollout_steps < self._cfg.num_rollout_steps and not trial_done:
                with torch.no_grad():
                    value = model.value_network(obs)
                    action_dist, _ = model.policy_network(obs)
                    log_prob = action_dist.log_prob(action)
                    values.append(value.squeeze(0).cpu())
                    log_probs.append(log_prob.mean().cpu())

                # Add sample to rollout replay buffer
                rollout_buffer.add(observation=obs, action=action, reward=reward, done=done)

            # Save episode reward i.e., number of total steps for an episode
            step += 1
            total_reward += reward
            if trial_done:
                episode_rewards.append(total_reward)
                total_reward = 0

            # Produce sample for training task
            if step % self._cfg.num_rollout_steps == 0 or trial_done:
                if rollout_buffer.num_total > 1:
                    with torch.no_grad():
                        next_value = model.value_network(obs)
                        next_value = next_value.squeeze(0).cpu()

                    observations = rollout_buffer.observations[: rollout_buffer.num_total]
                    actions = rollout_buffer.actions[: rollout_buffer.num_total]
                    dones = rollout_buffer.dones[: rollout_buffer.num_total]
                    rewards = rollout_buffer.rewards[: rollout_buffer.num_total]

                    # Updade running mean and variance
                    if model.state_normalizer is not None:
                        model.state_normalizer.update(observations)

                    # Done vector for computing GAE
                    dones_gae = dones.roll(-1).clone()
                    dones_gae[-1] = done

                    advs = self.compute_gae(
                        rewards=rewards,
                        values=torch.hstack(values),
                        dones=dones_gae,
                        next_value=next_value,
                        gamma=self._cfg.discount_factor,
                        lambda_=self._cfg.lambda_gae,
                    )

                    sample_producer_session.produce_sample(
                        (observations, actions, advs, values, log_probs, current_players, episode_rewards)
                    )
                else:
                    sample_producer_session.produce_sample((None, None, None, None, None, None, episode_rewards))

                # Reset the rollout
                rollout_buffer.reset()
                values = []
                log_probs = []
                current_players = []
            if trial_done:
                break

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish the model"""
        model_id = f"{run_session.run_id}_model"
        self.model_id = model_id
        observation_space = self._environment_specs.get_observation_space()
        action_space = self._environment_specs.get_action_space()

        # Initalize model
        self.model.model_id = model_id  # pylint: disable=attribute-defined-outside-init
        serialized_model = APPOModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(name=model_id, model=serialized_model)
        self.model.policy_network.to(self._torch_device)
        self.model.value_network.to(self._torch_device)

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
            hill_type="none",
        )
        replay_buffer = PPOReplayBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=(utils.flatdim(observation_space.gym_space),),
            action_shape=(utils.flatdim(action_space.gym_space),),
            device=self._torch_device,
            dtype=self._dtype,
            seed=self._cfg.seed,
        )

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx: int, iter_idx: int):
            agent_actor_params = cogment.ActorParameters(
                cog_settings,
                name="player",
                class_name=PLAYER_ACTOR_CLASS,
                implementation="actors.appo.APPOActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                    model_id=model_id,
                    model_iteration=iteration_info.iteration,
                    seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
                    model_update_frequency=self._cfg.num_rollout_steps,
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
        episode_rewards = []

        tot_num_updates = self._cfg.max_training_steps // (self._cfg.epoch_num_trials * self._cfg.num_rollout_steps)
        num_updates = 1
        total_steps = 0
        sample_size = 1
        ref_clipping = self._cfg.clipping_coef
        for iter_idx in range(self._cfg.num_iter):
            trials_id_and_params = [
                (f"{run_session.run_id}_{iter_idx}_{trial_idx}", create_trial_params(trial_idx, iter_idx))
                for trial_idx in range(self._cfg.epoch_num_trials)
            ]
            for step_idx, _, _, sample in run_session.start_and_await_trials(
                trials_id_and_params, self.sample_producer_impl, self._cfg.num_parallel_trials
            ):
                # Collect the rollout
                (trial_obs, trial_act, trial_adv, trial_val, trial_log_prob, _, trial_eps_rew) = sample
                episode_rewards.extend(trial_eps_rew)

                # Save data to replay buffer
                if trial_act is not None:
                    sample_size += len(trial_obs)
                    total_steps += len(trial_obs)
                    replay_buffer.add_multi_samples(
                        trial_obs=trial_obs,
                        trial_act=trial_act,
                        trial_adv=trial_adv,
                        trial_val=trial_val,
                        trial_log_prob=trial_log_prob,
                    )
                if (
                    replay_buffer.size() >= self._cfg.epoch_num_trials * self._cfg.num_rollout_steps
                    and step_idx % self._cfg.update_freq == 0
                    and sample_size >= self._cfg.epoch_num_trials * self._cfg.num_rollout_steps
                ):
                    sample_size = 1

                    # Get sample
                    data = replay_buffer.sample(self._cfg.epoch_num_trials * self._cfg.num_rollout_steps)

                    # Learning rate annealing
                    decaying_coef = 1.0 - (num_updates - 1.0) / tot_num_updates
                    curr_lr = self._cfg.learning_rate
                    curr_lr = max(decaying_coef * self._cfg.learning_rate, 1e-6)
                    self.policy_optimizer.param_groups[0]["lr"] = curr_lr
                    self.value_optimizer.param_groups[0]["lr"] = curr_lr

                    # Update parameters for policy and value networks
                    self.model.policy_network.to(self._torch_device)
                    self.model.value_network.to(self._torch_device)
                    policy_loss, value_loss = self.update_parameters(
                        observations=data.observation,
                        actions=data.action,
                        advs=data.adv,
                        values=data.value,
                        log_probs=data.log_prob,
                        num_epochs=self._cfg.num_epochs,
                        num_updates=num_updates,
                    )

                    # Compute the average reward i.e., average step length
                    avg_rewards = torch.zeros(1, dtype=self._dtype)
                    if len(episode_rewards) > 0:
                        avg_rewards = await self.compute_average_reward(episode_rewards)

                    # Send metric to mlflow
                    num_updates += 1
                    run_session.log_metrics(
                        model_iteration=iteration_info.iteration,
                        policy_loss=policy_loss.item(),
                        value_loss=value_loss.item(),
                        avg_rewards=avg_rewards.item(),
                        num_steps=total_steps,
                        num_updates=num_updates,
                        learning_rate=curr_lr,
                        clipping_coef=self._cfg.clipping_coef,
                    )
                    if num_updates % self._cfg.logging_interval == 0:
                        log.info(f"Steps: #{total_steps} | Avg. reward: {avg_rewards.item():.2f}")

                    # Publish the newly updated model
                    self.model.iter_idx = iter_idx
                    serialized_model = APPOModel.serialize_model(self.model)

                    if num_updates % 50 == 0:
                        iteration_info = await run_session.model_registry.store_model(
                            name=model_id, model=serialized_model
                        )
                    else:
                        iteration_info = await run_session.model_registry.publish_model(
                            name=model_id, model=serialized_model
                        )

                    self.model.policy_network.to(self._torch_device)
                    self.model.value_network.to(self._torch_device)
            if total_steps > self._cfg.max_training_steps:
                break
        iteration_info = await run_session.model_registry.store_model(name=model_id, model=serialized_model)

    def update_parameters(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advs: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        num_epochs: int,
        num_updates: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update policy & value networks"""

        returns = advs + values
        num_obs = len(returns)
        global_idx = np.arange(num_obs)
        for i in range(num_epochs):
            np.random.seed(self._cfg.seed + i + num_updates)
            np.random.shuffle(global_idx)
            for i in range(0, num_obs, self._cfg.batch_size):
                # Get data in batch
                idx = global_idx[i : i + self._cfg.batch_size]
                if len(idx) < self._cfg.batch_size:
                    break
                observation = observations[idx]
                action = actions[idx]
                return_ = returns[idx]
                adv = advs[idx].clone()
                old_value = values[idx]
                old_log_prob = log_probs[idx]

                # Normalize the observation
                if self.model.state_normalizer is not None:
                    observation = self.model.state_normalizer.normalize(observation)

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Value loss
                value = self.model.value_network(observation)
                value_loss_unclipped = (value - return_) ** 2
                value_loss_max = value_loss_unclipped
                if self._cfg.is_vf_clipped:
                    value_clipped = old_value + torch.clamp(
                        value - old_value, -self._cfg.clipping_coef, self._cfg.clipping_coef
                    )
                    value_loss_clipped = (value_clipped - return_) ** 2
                    value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss = 0.5 * value_loss_max.mean() * self._cfg.value_loss_coef

                # Policy loss
                action_dist, _ = self.model.policy_network(observation)
                new_log_prob = action_dist.log_prob(action).mean(axis=1)
                entropy = action_dist.entropy()
                ratio = torch.exp(new_log_prob.view(-1, 1) - old_log_prob)

                policy_loss_1 = -adv * ratio
                policy_loss_2 = -adv * torch.clamp(ratio, 1 - self._cfg.clipping_coef, 1 + self._cfg.clipping_coef)

                entropy_loss = entropy.mean()
                policy_loss = (
                    torch.max(policy_loss_1, policy_loss_2).mean() - self._cfg.entropy_loss_coef * entropy_loss
                )

                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.value_network.parameters(), self._cfg.grad_norm)
                self.value_optimizer.step()

                # Udapte policy network
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.policy_network.parameters(), self._cfg.grad_norm)
                self.policy_optimizer.step()

        return policy_loss, value_loss

    @staticmethod
    async def compute_average_reward(rewards: list) -> torch.Tensor:
        """Compute the average reward of the last 100 episode"""
        last_100_rewards = rewards[np.maximum(0, len(rewards) - 100) : len(rewards)]
        return torch.vstack(last_100_rewards).mean()

    def normalize_rewards(self, rewards: list, model: APPOModel) -> list:
        """Normalize the rewards"""
        normalized_reward = []
        for rew in rewards:
            normalized_reward.append(rew / (model.reward_normalizaiton.var + 1e-8) ** 0.5)
            self.returns = self.returns * self._cfg.discount_factor + rew
            model.reward_normalization.update(self.returns)

        return normalized_reward

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ):
        """Compute Generalized Advantage Estimation (GAE). See equations 11 & 12 in
        https://arxiv.org/pdf/1707.06347.pdf
        """
        advs = []
        with torch.no_grad():
            gae = 0.0
            # dones = torch.cat((dones, torch.zeros(1, 1).to(self._torch_device)), dim=0)
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
                gae = delta + gamma * lambda_ * (1 - dones[i]) * gae
                advs.append(gae)
                next_value = values[i]
        advs.reverse()

        return advs
