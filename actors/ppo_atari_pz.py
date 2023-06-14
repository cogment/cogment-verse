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
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import cogment
import numpy as np
import torch
from cogment.actor import ActorSession
from gymnasium.spaces import Discrete, utils
from omegaconf import DictConfig, ListConfig
from torch.distributions.distribution import Distribution

from cogment_verse import HumanDataBuffer, Model, PPOReplayBuffer, RolloutBuffer
from cogment_verse.constants import ActorSpecType
from cogment_verse.run.run_session import RunSession
from cogment_verse.run.sample_producer_worker import SampleProducerSession
from cogment_verse.specs import (
    EVALUATOR_ACTOR_CLASS,
    HUMAN_ACTOR_IMPL,
    PLAYER_ACTOR_CLASS,
    WEB_ACTOR_NAME,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentSpecs,
    cog_settings,
)

torch.multiprocessing.set_sharing_strategy("file_system")


log = logging.getLogger(__name__)


# pylint: disable=protected-access
# pylint: disable=abstract-method
# pylint: disable=unused-argument
# pylint: disable=unused-variable
# pylint: disable=too-many-lines
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


class PolicyValueNetwork(torch.nn.Module):
    """Policy and Value networks for Atari games"""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.shared_network = torch.nn.Sequential(
            initialize_layer(torch.nn.Conv2d(6, 32, 8, stride=4)),
            torch.nn.ReLU(),
            initialize_layer(torch.nn.Conv2d(32, 64, 3, stride=2)),
            torch.nn.ReLU(),
            initialize_layer(torch.nn.Conv2d(64, 64, 3, stride=1)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            initialize_layer(torch.nn.Linear(64 * 7 * 7, 512)),
            torch.nn.ReLU(),
        )
        self.actor = initialize_layer(torch.nn.Linear(512, num_actions), std=0.01)
        self.value = initialize_layer(torch.nn.Linear(512, 1), std=1)

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute the value of being in a state"""
        observation_clone = observation.clone()
        observation_clone = observation_clone / 255.0
        return self.value(self.shared_network(observation_clone))

    def get_action(self, observation: torch.Tensor) -> Distribution:
        """Actions given observations"""
        observation_clone = observation.clone()
        observation_clone = observation_clone / 255.0
        action_logits = self.actor(self.shared_network(observation_clone))
        dist = torch.distributions.categorical.Categorical(logits=action_logits)

        return dist

    def get_action_value(self, observation: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get value and log prob"""
        observation_clone = observation.clone()
        observation_clone = observation_clone / 255.0
        hidden = self.shared_network(observation_clone)

        # Log probs
        action_logits = self.actor(hidden)
        dist = torch.distributions.categorical.Categorical(logits=action_logits)
        log_probs = dist.log_prob(action)

        # Value
        values = self.value(hidden)
        return values, log_probs, dist.entropy()


class PPOModel(Model):
    """Proximal Policy Optimization (PPO) is an on-policy algorithm.
    https://arxiv.org/pdf/1707.06347.pdf.

    Attributes:

    """

    def __init__(
        self,
        model_id: int,
        environment_implementation: str,
        num_actions: int,
        input_shape: tuple,
        num_policy_outputs: int,
        n_iter: int = 1000,
        dtype=torch.float32,
        device: str = "cpu",
        iteration: int = 0,
    ) -> None:
        super().__init__(model_id, iteration)
        self._environment_implementation = environment_implementation
        self._num_actions = num_actions
        self.input_shape = input_shape
        self.num_policy_outputs = num_policy_outputs
        self.device = device
        self._dtype = dtype
        self._n_iter = n_iter
        self.network = PolicyValueNetwork(num_actions=self._num_actions).to(self._dtype)
        self.network.to(torch.device(self.device))

        # version user data
        self.iter_idx = 0
        self.total_samples = 0

    def get_model_user_data(self) -> dict:
        """Get user model"""
        return {
            "model_id": self.model_id,
            "environment_implementation": self._environment_implementation,
            "num_actions": self._num_actions,
            "input_shape": self.input_shape,
            "num_policy_outputs": self.num_policy_outputs,
            "device": self.device,
            "iter_idx": self.iter_idx,
            "total_samples": self.total_samples,
        }

    @staticmethod
    def serialize_model(model) -> bytes:
        stream = io.BytesIO()
        torch.save(
            (
                model.network.cpu().state_dict(),
                model.get_model_user_data(),
            ),
            stream,
        )
        return stream.getvalue()

    @classmethod
    def deserialize_model(cls, serialized_model) -> PPOModel:
        stream = io.BytesIO(serialized_model)
        (network_state_dict, model_user_data) = torch.load(stream)

        model = PPOModel(
            model_id=model_user_data["model_id"],
            environment_implementation=model_user_data["environment_implementation"],
            num_actions=model_user_data["num_actions"],
            input_shape=model_user_data["input_shape"],
            num_policy_outputs=model_user_data["num_policy_outputs"],
            device="cpu",
        )
        model.network.load_state_dict(network_state_dict)
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

    async def impl(self, actor_session: ActorSession):
        # Start a session
        actor_session.start()
        config = actor_session.config

        # Setup random seed
        torch.manual_seed(config.seed)

        # Get observation and action space
        spec_type = ActorSpecType.from_config(config.spec_type)
        actor_specs = EnvironmentSpecs.deserialize(config.environment_specs)[spec_type]
        observation_space = actor_specs.get_observation_space()
        action_space = actor_specs.get_action_space(seed=config.seed)

        # Get model
        model = await PPOModel.retrieve_model(actor_session.model_registry, config.model_id, config.model_iteration)
        model.network.eval()

        log.info(f"Actor - retrieved model number: {model.iteration}")
        obs_shape = model.input_shape[::-1]

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                if (
                    event.observation.observation.HasField("current_player")
                    and event.observation.observation.current_player.name != actor_session.name
                ):
                    # Not the turn of the agent
                    actor_session.do_action(action_space.serialize(action_space.create()))
                    continue

                # Retrieve the model every N rollout steps
                if (
                    config.model_update_frequency != 0
                    and (event.observation.tick_id + 1) % config.model_update_frequency == 0
                ):
                    model = await PPOModel.retrieve_model(actor_session.model_registry, config.model_id, -1)
                    model.network.eval()
                    # model.network.to(torch.device(model.device))

                obs = observation_space.deserialize(event.observation.observation)
                obs_tensor = torch.tensor(obs.flat_value, dtype=self._dtype).reshape(obs_shape).clone()
                obs_tensor = torch.unsqueeze(obs_tensor.permute((2, 0, 1)), dim=0)

                # Get action from policy network
                with torch.no_grad():
                    dist = model.network.get_action(obs_tensor)
                    action = dist.sample().cpu().numpy()[0]

                # Send action to environment
                assert isinstance(action_space.gym_space, Discrete)  # TODO: test with other action space types
                action_value = action_space.create(value=action)
                actor_session.do_action(action_space.serialize(action_value))


class BasePPOTraining(ABC):
    """Abstract class for training PPO agent"""

    default_cfg = {
        "seed": 0,
        "num_epochs": 10,
        "num_iter": 500,
        "epoch_num_trials": 1,
        "num_parallel_trials": 1,
        "discount_factor": 0.99,
        "entropy_loss_coef": 0.05,
        "value_loss_coef": 0.5,
        "clipping_coef": 0.1,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "buffer_size": 1000,
        "learning_starts": 128,
        "update_freq": 1,
        "num_rollout_steps": 10,
        "lambda_gae": 0.95,
        "device": "cpu",
        "grad_norm": 0.5,
        "image_size": [6, 84, 84],
        "logging_interval": 100,
    }

    def __init__(self, environment_specs: EnvironmentSpecs, cfg: Union[ListConfig, DictConfig]) -> None:
        super().__init__()
        self._dtype = torch.float32
        self._environment_specs = environment_specs
        self._cfg = cfg
        self._spec_type = ActorSpecType.DEFAULT
        available_device = get_device(self._cfg.device)
        self._torch_device = torch.device(available_device)
        self.returns = 0
        self.model_id = ""

        # Set random seed for initializing neural network parameters
        torch.manual_seed(self._cfg.seed)
        self.model = PPOModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            num_actions=utils.flatdim(self._environment_specs[self._spec_type].get_action_space().gym_space),
            input_shape=tuple(self._cfg.image_size),
            num_policy_outputs=1,
            n_iter=self._cfg.num_epochs,
            device=available_device,
            dtype=self._dtype,
        )

        # Get optimizer for two models
        self.network_optimizer = torch.optim.Adam(self.model.network.parameters(), lr=self._cfg.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.network_optimizer, step_size=1000, gamma=0.1)

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
        player_observation_space = player_environment_specs[self._spec_type].get_observation_space()
        player_action_space = player_environment_specs[self._spec_type].get_action_space()
        num_players = player_environment_specs.num_players
        obs_shape = tuple(self._cfg.image_size)[::-1]

        rollout_buffer = RolloutBuffer(
            capacity=self._cfg.num_rollout_steps,
            observation_shape=self._cfg.image_size,
            action_shape=(1,),
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
            previous_actor_sample = sample.actors_data[player_actor_name]
            player_actor_name = previous_actor_sample.observation.current_player.name
            actor_sample = sample.actors_data[player_actor_name]

            if self.should_load_model(sample.tick_id, self._cfg.num_rollout_steps, trial_done):
                model = await PPOModel.retrieve_model(sample_producer_session.model_registry, self.model_id, -1)
                model.network.eval()

            # Collect data from environment
            obs_flat = player_observation_space.deserialize(actor_sample.observation).flat_value
            obs_tensor = torch.tensor(obs_flat, dtype=self._dtype).reshape(obs_shape).clone()
            observation_value = torch.unsqueeze(torch.permute(obs_tensor, (2, 0, 1)), dim=0)
            done = torch.ones(1, dtype=torch.float32) if trial_done else torch.zeros(1, dtype=torch.float32)
            reward_value = (
                torch.tensor(actor_sample.reward, dtype=self._dtype)
                if actor_sample.reward is not None
                else torch.tensor(0, dtype=self._dtype)
            )

            if not trial_done:
                action_value = torch.tensor(
                    player_action_space.deserialize(actor_sample.action).value, dtype=self._dtype
                )
                current_players.append(player_actor_name)

            # Compute values and log probs
            if step % self._cfg.num_rollout_steps > 0 and not trial_done:
                with torch.no_grad():
                    value, log_prob, _ = model.network.get_action_value(
                        observation=observation_value, action=action_value
                    )
                    values.append(value.squeeze(0).cpu())
                    log_probs.append(log_prob.squeeze(0).cpu())

                # Add sample to rollout replay buffer
                rollout_buffer.add(observation=observation_value, action=action_value, reward=reward_value, done=done)

            # Save episode reward i.e., number of total steps for an episode
            step += 1
            total_reward += 1
            if trial_done:
                episode_rewards.append(torch.tensor(total_reward / num_players, dtype=self._dtype))
                total_reward = 0

            # Produce sample for training task
            if step % self._cfg.num_rollout_steps == 0 or trial_done:
                if rollout_buffer.num_total > 1:
                    with torch.no_grad():
                        next_value = model.network.get_value(observation_value)
                        next_value = next_value.squeeze(0).cpu()
                    advs = self.compute_gae(
                        rewards=rollout_buffer.rewards[: rollout_buffer.num_total],
                        values=torch.hstack(values),
                        dones=rollout_buffer.dones[: rollout_buffer.num_total],
                        next_value=next_value,
                        gamma=self._cfg.discount_factor,
                        lambda_=self._cfg.lambda_gae,
                    )
                    observations = rollout_buffer.observations[: rollout_buffer.num_total]
                    actions = rollout_buffer.actions[: rollout_buffer.num_total]
                    sample_producer_session.produce_sample(
                        (observations, actions, advs, values, log_probs, current_players, episode_rewards)
                    )
                else:
                    sample_producer_session.produce_sample((None, None, None, None, None, None, episode_rewards))

                # Reset the rollout
                rollout_buffer.reset()
                values = []
                log_probs = []
            if trial_done:
                break

    @abstractmethod
    async def impl(self, run_session: RunSession) -> dict:
        raise NotImplementedError

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

                adv = (adv - adv.mean()) / (adv.std() + 1e-6)

                # Compute the value and values loss
                value, new_log_prob, entropy = self.model.network.get_action_value(
                    observation=observation, action=action.long().flatten()
                )
                value_loss_unclipped = (value - return_) ** 2
                value_clipped = old_value + torch.clamp(value - old_value, -0.1, 0.1)
                value_loss_clipped = (value_clipped - return_) ** 2
                value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss = 0.5 * value_loss_max.mean()

                # Get action distribution & the log-likelihood
                ratio = torch.exp(new_log_prob.view(-1, 1) - old_log_prob)

                # Compute policy loss
                policy_loss_1 = -adv * ratio
                policy_loss_2 = -adv * torch.clamp(ratio, 1 - self._cfg.clipping_coef, 1 + self._cfg.clipping_coef)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Loss
                entropy_loss = entropy.mean()
                loss = policy_loss - self._cfg.entropy_loss_coef * entropy_loss + value_loss * self._cfg.value_loss_coef

                # Update value network
                self.network_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), self._cfg.grad_norm)
                self.network_optimizer.step()

        return policy_loss, value_loss

    @staticmethod
    async def compute_average_reward(rewards: list) -> torch.Tensor:
        """Compute the average reward of the last 100 episode"""
        last_100_rewards = rewards[np.maximum(0, len(rewards) - 100) : len(rewards)]
        return torch.vstack(last_100_rewards).mean()

    def compute_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute values given the states"""
        values = []
        with torch.no_grad():
            for obs in observations:
                values.append(self.model.network.get_value(obs))

        return torch.vstack(values)

    def compute_batch_log_lik(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood for each actions"""
        log_probs = []
        for obs, action in zip(observations, actions):
            log_prob = self.compute_log_lik(observation=obs, action=action)
            log_probs.append(log_prob)

        return torch.vstack(log_probs)

    def compute_log_lik(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute the log likelihood for each actions"""
        with torch.no_grad():
            dist = self.model.network.get_action(observation)
        log_prob = dist.log_prob(action.flatten()).view(-1, 1)

        return log_prob

    @staticmethod
    async def get_n_steps_data(dataset: torch.Tensor, num_steps: int) -> torch.tensor:
        """Get the data up to nth steps"""
        return dataset[:num_steps]

    def make_dataloader(self, dataset: torch.Tensor, batch_size: int, obs_shape: int) -> List[torch.Tensor]:
        """Create a dataloader in batches"""
        # Initialization
        output_batches = torch.zeros((batch_size, *obs_shape), dtype=self._dtype).to(self._torch_device)
        num_data = len(dataset)
        data_loader = []
        count = 0
        for i, y_batch in enumerate(dataset):
            output_batches[count] = y_batch
            # Store data
            if (i + 1) % batch_size == 0:
                data_loader.append(output_batches)

                # Reset
                count = 0
                output_batches = torch.zeros((batch_size, *obs_shape), dtype=self._dtype).to(self._torch_device)
            else:
                count += 1
                if i == num_data - 1:
                    data_loader.append(output_batches[:count])

        return data_loader

    def normalize_rewards(self, rewards: list, model: PPOModel) -> list:
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


class PPOSelfTraining(BasePPOTraining):
    """Train PPO agent"""

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish the model"""
        model_id = f"{run_session.run_id}_model"
        self.model_id = model_id

        # Initalize model
        self.model.model_id = model_id  # pylint: disable=attribute-defined-outside-init
        serialized_model = PPOModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(name=model_id, model=serialized_model)
        self.model.network.to(self._torch_device)

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
            hill_type="none",
        )

        replay_buffer = PPOReplayBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=self._cfg.image_size,
            action_shape=(1,),
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
                implementation="actors.ppo_atari_pz.PPOActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                    spec_type=self._spec_type.value,
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
        for iter_idx in range(self._cfg.num_iter):
            trials_id_and_params = [
                (f"{run_session.run_id}_{iter_idx}_{trial_idx}", create_trial_params(trial_idx, iter_idx))
                for trial_idx in range(self._cfg.epoch_num_trials)
            ]
            for step_idx, trial_id, trial_idx, sample in run_session.start_and_await_trials(
                trials_id_and_params, self.sample_producer_impl, self._cfg.num_parallel_trials
            ):
                # Collect the rollout
                (trial_obs, trial_act, trial_adv, trial_val, trial_log_prob, _, trial_eps_rew) = sample
                episode_rewards.extend(trial_eps_rew)

                # Save data to replay buffer
                if trial_act is not None:
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
                ):
                    # Get sample
                    data = replay_buffer.sample(self._cfg.epoch_num_trials * self._cfg.num_rollout_steps)

                    # Learning rate annealing
                    decaying_coef = 1.0 - (num_updates - 1.0) / tot_num_updates
                    curr_lr = decaying_coef * self._cfg.learning_rate
                    self.network_optimizer.param_groups[0]["lr"] = curr_lr

                    # Update parameters for policy and value networks
                    self.model.network.to(self._torch_device)
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
                    total_steps += self._cfg.epoch_num_trials * self._cfg.num_rollout_steps
                    num_updates += 1
                    run_session.log_metrics(
                        model_iteration=iteration_info.iteration,
                        policy_loss=policy_loss.item(),
                        value_loss=value_loss.item(),
                        avg_rewards=avg_rewards.item(),
                        num_steps=total_steps,
                        num_updates=num_updates,
                    )
                    if num_updates % self._cfg.logging_interval == 0:
                        log.info(f"Steps: #{total_steps} | Avg. reward: {avg_rewards.item():.2f}")

                    # Publish the newly updated model
                    self.model.iter_idx = iter_idx
                    serialized_model = PPOModel.serialize_model(self.model)

                    if num_updates % 50 == 0:
                        iteration_info = await run_session.model_registry.store_model(
                            name=model_id, model=serialized_model
                        )
                    else:
                        iteration_info = await run_session.model_registry.publish_model(
                            name=model_id, model=serialized_model
                        )

                    self.model.network.to(self._torch_device)
            if total_steps > self._cfg.max_training_steps:
                break
        iteration_info = await run_session.model_registry.store_model(name=model_id, model=serialized_model)


class HillPPOTraining(BasePPOTraining):
    """Train PPO agent using human's actions"""

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish the model"""
        model_id = f"{run_session.run_id}_model"
        self.model_id = model_id

        # Initalize model
        self.model.model_id = model_id  # pylint: disable=attribute-defined-outside-init
        serialized_model = PPOModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(name=model_id, model=serialized_model)
        self.model.network.to(self._torch_device)

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
            hill_type="demo",
        )

        # Human data buffer
        human_data_category = "demo"
        human_data_buffer = HumanDataBuffer(
            observation_shape=tuple(self._cfg.image_size),
            action_shape=(1,),
            human_data_category=human_data_category,
            capacity=self._cfg.buffer_size,
            action_dtype=np.int32,
            file_name=f"{human_data_category}_{run_session.run_id}",
            seed=self._cfg.seed,
        )
        # TODO: merge buffer size
        replay_buffer = PPOReplayBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=self._cfg.image_size,
            action_shape=(1,),
            device=self._torch_device,
            dtype=self._dtype,
            seed=self._cfg.seed,
        )

        # Create actor parameters
        def create_actor_params(
            actor_names: List[str], trial_idx: int, iter_idx: int, hill_training_trial_period: int, iteration: int = -1
        ):
            np.random.default_rng(self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials)
            human_actor_idx = np.random.choice(len(actor_names), 1, replace=False)
            log.info(f"human idx {human_actor_idx}")
            human = True
            actors = []
            for i, name in enumerate(actor_names):
                if human and i == human_actor_idx[0]:
                    actor = cogment.ActorParameters(
                        cog_settings,
                        name=WEB_ACTOR_NAME,
                        class_name=PLAYER_ACTOR_CLASS,
                        implementation=HUMAN_ACTOR_IMPL,
                        config=AgentConfig(
                            run_id=run_session.run_id,
                            environment_specs=self._environment_specs.serialize(),
                            spec_type=self._spec_type.value,
                            model_iteration=iteration_info.iteration,
                            seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
                        ),
                    )
                else:
                    actor = cogment.ActorParameters(
                        cog_settings,
                        name=name,
                        class_name=PLAYER_ACTOR_CLASS,
                        implementation="actors.ppo_atari_pz.PPOActor",
                        config=AgentConfig(
                            run_id=run_session.run_id,
                            environment_specs=self._environment_specs.serialize(),
                            spec_type=self._spec_type.value,
                            model_id=model_id,
                            model_iteration=iteration_info.iteration,
                            seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
                            model_update_frequency=self._cfg.num_rollout_steps,
                        ),
                    )
                actors.append(actor)

            return actors

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx: int, iter_idx: int, actors: list):
            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id,
                    render=HUMAN_ACTOR_IMPL in [actor.implementation for actor in actors],
                    seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
                ),
                actors=actors,
            )

        hill_training_trial_period = (
            math.floor(1 / self._cfg.hill_training_trials_ratio) if self._cfg.hill_training_trials_ratio > 0 else 0
        )

        # Run environment
        episode_rewards = []
        tot_num_updates = self._cfg.max_training_steps // (self._cfg.epoch_num_trials * self._cfg.num_rollout_steps)

        num_updates = 0
        total_steps = 0
        for iter_idx in range(self._cfg.num_iter):
            log.info(f"iteration {iter_idx}")
            # TODO: actor names should not be hard-coding
            trials_id_and_params = [
                (
                    f"{run_session.run_id}_{iter_idx}_{trial_idx}",
                    create_trial_params(
                        trial_idx,
                        iter_idx,
                        actors=create_actor_params(
                            actor_names=["first_0", "second_0"],
                            trial_idx=trial_idx,
                            iter_idx=iter_idx,
                            hill_training_trial_period=hill_training_trial_period,
                        ),
                    ),
                )
                for trial_idx in range(self._cfg.epoch_num_trials)
            ]
            for step_idx, _, _, sample in run_session.start_and_await_trials(
                trials_id_and_params, self.sample_producer_impl, self._cfg.num_parallel_trials
            ):
                # Collect the rollout
                (trial_obs, trial_act, trial_adv, trial_val, trial_log_prob, trial_act_name, trial_eps_rew) = sample
                episode_rewards.extend(trial_eps_rew)

                # Save data to replay buffer
                if trial_act is not None:
                    replay_buffer.add_multi_samples(
                        trial_obs=trial_obs,
                        trial_act=trial_act,
                        trial_adv=trial_adv,
                        trial_val=trial_val,
                        trial_log_prob=trial_log_prob,
                    )

                    # Humand demo data
                    trial_human_obs = [
                        obs for (obs, actor_name) in zip(trial_obs, trial_act_name) if actor_name == WEB_ACTOR_NAME
                    ]
                    trial_human_act = [
                        act for (act, actor_name) in zip(trial_act, trial_act_name) if actor_name == WEB_ACTOR_NAME
                    ]
                    human_data_buffer.add_multi_samples(trial_obs=trial_human_obs, trial_act=trial_human_act)

                if (
                    replay_buffer.size() >= self._cfg.epoch_num_trials * self._cfg.num_rollout_steps
                    and step_idx % self._cfg.update_freq == 0
                ):
                    # Get sample
                    data = replay_buffer.sample(self._cfg.epoch_num_trials * self._cfg.num_rollout_steps)

                    # Learning rate annealing
                    decaying_coef = 1.0 - (num_updates - 1.0) / tot_num_updates
                    curr_lr = decaying_coef * self._cfg.learning_rate
                    self.network_optimizer.param_groups[0]["lr"] = curr_lr

                    # Update parameters for policy and value networks
                    self.model.network.to(self._torch_device)
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
                    total_steps += self._cfg.epoch_num_trials * self._cfg.num_rollout_steps
                    num_updates += 1
                    run_session.log_metrics(
                        model_iteration=iteration_info.iteration,
                        policy_loss=policy_loss.item(),
                        value_loss=value_loss.item(),
                        avg_rewards=avg_rewards.item(),
                        num_steps=total_steps,
                        num_updates=num_updates,
                    )
                    if num_updates % self._cfg.logging_interval == 0:
                        log.info(f"Steps: #{total_steps} | Avg. reward: {avg_rewards.item():.2f}")

                    # Publish the newly updated model
                    self.model.iter_idx = iter_idx
                    serialized_model = PPOModel.serialize_model(self.model)

                    if num_updates % 50 == 0:
                        iteration_info = await run_session.model_registry.store_model(
                            name=model_id, model=serialized_model
                        )
                    else:
                        iteration_info = await run_session.model_registry.publish_model(
                            name=model_id, model=serialized_model
                        )

                    self.model.network.to(self._torch_device)
            if total_steps > self._cfg.max_training_steps:
                break
        iteration_info = await run_session.model_registry.store_model(name=model_id, model=serialized_model)


class HumanFeedbackPPOTraining(BasePPOTraining):
    """Train PPO agent with human feedback"""

    async def sample_producer_impl(self, sample_producer_session: SampleProducerSession):
        """Collect sample from the trial"""
        # Load model
        model = await PPOModel.retrieve_model(sample_producer_session.model_registry, self.model_id, -1)
        model.network.eval()

        # Actor parameter
        actor_params = {
            actor_params.name: actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == PLAYER_ACTOR_CLASS
        }
        actor_names = list(actor_params.keys())
        player_environment_specs = EnvironmentSpecs.deserialize(actor_params[actor_names[0]].config.environment_specs)
        player_observation_space = player_environment_specs[self._spec_type].get_observation_space()
        player_action_space = player_environment_specs[self._spec_type].get_action_space()
        num_players = player_environment_specs.num_players
        obs_shape = tuple(self._cfg.image_size)[::-1]

        # Rollout buffer
        rollout_buffer = RolloutBuffer(
            capacity=self._cfg.num_rollout_steps,
            observation_shape=self._cfg.image_size,
            action_shape=(1,),
            action_dtype=torch.float32,
        )

        values = []
        log_probs = []
        human_eval_scores = []
        current_players = []
        episode_rewards = []
        player_actor_name = actor_names[0]
        wait_for_feedback = False
        step = 0
        total_reward = 0
        async for sample in sample_producer_session.all_trial_samples():
            # Trail status
            trial_done = sample.trial_state == cogment.TrialState.ENDED

            # Load model
            if self.should_load_model(sample.tick_id, self._cfg.num_rollout_steps, trial_done):
                model = await PPOModel.retrieve_model(sample_producer_session.model_registry, self.model_id, -1)
                model.network.eval()

            # Actor names
            previous_actor_sample = sample.actors_data[player_actor_name]
            player_actor_name = previous_actor_sample.observation.current_player.name
            actor_sample = sample.actors_data[player_actor_name]

            # Collect data
            if player_actor_name != WEB_ACTOR_NAME:
                obs_flat = player_observation_space.deserialize(actor_sample.observation).flat_value
                obs_tensor = torch.tensor(obs_flat, dtype=self._dtype).reshape(obs_shape).clone()
                observation_value = torch.unsqueeze(torch.permute(obs_tensor, (2, 0, 1)), dim=0)
                done = torch.ones(1, dtype=torch.float32) if trial_done else torch.zeros(1, dtype=torch.float32)
                reward_value = (
                    torch.tensor(actor_sample.reward, dtype=self._dtype)
                    if actor_sample.reward is not None
                    else torch.tensor(0, dtype=self._dtype)
                )
                if not trial_done:
                    action_value = torch.tensor(
                        player_action_space.deserialize(actor_sample.action).value, dtype=self._dtype
                    )
                current_players.append(player_actor_name)
                wait_for_feedback = True
            else:
                wait_for_feedback = False
                human_eval_score = torch.tensor(
                    actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype
                )
                human_eval_scores.append(human_eval_score)

            # Compute values and log probs
            if (
                step % self._cfg.num_rollout_steps < self._cfg.num_rollout_steps and not trial_done
            ) and not wait_for_feedback:
                with torch.no_grad():
                    value, log_prob, _ = model.network.get_action_value(
                        observation=observation_value, action=action_value
                    )
                    values.append(value.squeeze(0).cpu())
                    log_probs.append(log_prob.squeeze(0).cpu())

                # Add sample to rollout replay buffer
                combined_reward = reward_value + human_eval_score
                rollout_buffer.add(
                    observation=observation_value, action=action_value, reward=combined_reward, done=done
                )

            # Save episode reward i.e., number of total steps for an episode
            step += 1
            total_reward += 1
            if trial_done:
                episode_rewards.append(torch.tensor(total_reward / num_players, dtype=self._dtype))
                total_reward = 0

            # Produce sample for training task
            if (step % self._cfg.num_rollout_steps == 0 or trial_done) and not wait_for_feedback:
                if rollout_buffer.num_total > 1:
                    with torch.no_grad():
                        next_value = model.network.get_value(observation_value)
                        next_value = next_value.squeeze(0).cpu()
                    advs = self.compute_gae(
                        rewards=rollout_buffer.rewards[: rollout_buffer.num_total],
                        values=torch.hstack(values),
                        dones=rollout_buffer.dones[: rollout_buffer.num_total],
                        next_value=next_value,
                        gamma=self._cfg.discount_factor,
                        lambda_=self._cfg.lambda_gae,
                    )
                    observations = rollout_buffer.observations[: rollout_buffer.num_total]
                    actions = rollout_buffer.actions[: rollout_buffer.num_total]
                    sample_producer_session.produce_sample(
                        (
                            observations,
                            actions,
                            advs,
                            values,
                            log_probs,
                            human_eval_scores,
                            current_players,
                            episode_rewards,
                        )
                    )
                else:
                    sample_producer_session.produce_sample((None, None, None, None, None, None, None, episode_rewards))

                # Reset the rollout
                rollout_buffer.reset()
                values = []
                log_probs = []
                human_eval_scores = []
            if trial_done:
                break

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish the model"""
        model_id = f"{run_session.run_id}_model"
        self.model_id = model_id

        # Initalize model
        self.model.model_id = model_id  # pylint: disable=attribute-defined-outside-init
        serialized_model = PPOModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )
        self.model.network.to(self._torch_device)

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
            hill_type="feedback",
        )

        # Human data buffer
        human_data_category = "feedback"
        human_data_buffer = HumanDataBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=tuple(self._cfg.image_size),
            action_shape=(1,),
            human_data_category=human_data_category,
            action_dtype=np.float32,
            file_name=f"{human_data_category}_{run_session.run_id}",
            seed=self._cfg.seed,
        )

        # APPO replay buffer
        replay_buffer = PPOReplayBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=self._cfg.image_size,
            action_shape=(1,),
            device=self._torch_device,
            dtype=self._dtype,
            seed=self._cfg.seed,
        )

        # Create actor parameters
        def create_actor_params(actor_names: List[str], trial_idx: int, iter_idx: int, hill_training_trial_period: int):
            actors = []
            for i, name in enumerate(actor_names):
                actor = cogment.ActorParameters(
                    cog_settings,
                    name=name,
                    class_name=PLAYER_ACTOR_CLASS,
                    implementation="actors.ppo_atari_pz.PPOActor",
                    config=AgentConfig(
                        run_id=run_session.run_id,
                        environment_specs=self._environment_specs.serialize(),
                        spec_type=self._spec_type.value,
                        model_id=model_id,
                        model_iteration=iteration_info.iteration,
                        seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
                        model_update_frequency=self._cfg.num_rollout_steps,
                    ),
                )
                actors.append(actor)

            actor = cogment.ActorParameters(
                cog_settings,
                name=WEB_ACTOR_NAME,
                class_name=EVALUATOR_ACTOR_CLASS,
                implementation=HUMAN_ACTOR_IMPL,
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                    spec_type=self._spec_type.value,
                ),
            )
            actors.append(actor)

            return actors

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx: int, iter_idx: int, actors: list):
            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id,
                    render=HUMAN_ACTOR_IMPL in [actor.implementation for actor in actors],
                    seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
                ),
                actors=actors,
            )

        hill_training_trial_period = (
            math.floor(1 / self._cfg.hill_training_trials_ratio) if self._cfg.hill_training_trials_ratio > 0 else 0
        )

        # Run environment
        episode_rewards = []
        tot_num_updates = self._cfg.max_training_steps // (self._cfg.epoch_num_trials * self._cfg.num_rollout_steps)

        num_updates = 0
        total_steps = 0
        for iter_idx in range(self._cfg.num_iter):
            # TODO: actor names should not be hard-coding
            trials_id_and_params = [
                (
                    f"{run_session.run_id}_{iter_idx}_{trial_idx}",
                    create_trial_params(
                        trial_idx,
                        iter_idx,
                        actors=create_actor_params(
                            actor_names=["first_0", "second_0"],
                            trial_idx=trial_idx,
                            iter_idx=iter_idx,
                            hill_training_trial_period=hill_training_trial_period,
                        ),
                    ),
                )
                for trial_idx in range(self._cfg.epoch_num_trials)
            ]

            # Run trial
            for step_idx, _, _, sample in run_session.start_and_await_trials(
                trials_id_and_params=trials_id_and_params,
                sample_producer_impl=self.sample_producer_impl,
                num_parallel_trials=self._cfg.num_parallel_trials,
            ):
                # Collect the rollout
                (trial_obs, trial_act, trial_adv, trial_val, trial_log_prob, trial_hb, _, trial_eps_rew) = sample
                episode_rewards.extend(trial_eps_rew)

                # Save data to replay buffer
                if trial_act is not None:
                    replay_buffer.add_multi_samples(
                        trial_obs=trial_obs,
                        trial_act=trial_act,
                        trial_adv=trial_adv,
                        trial_val=trial_val,
                        trial_log_prob=trial_log_prob,
                    )
                    human_data_buffer.add_multi_samples_with_hb(
                        trial_obs=trial_obs, trial_act=trial_act, trial_hb=trial_hb
                    )

                # Training
                if (
                    replay_buffer.size() >= self._cfg.epoch_num_trials * self._cfg.num_rollout_steps
                    and step_idx % self._cfg.update_freq == 0
                ):
                    # Get sample
                    data = replay_buffer.sample(self._cfg.epoch_num_trials * self._cfg.num_rollout_steps)

                    # Learning rate annealing
                    decaying_coef = 1.0 - (num_updates - 1.0) / tot_num_updates
                    curr_lr = decaying_coef * self._cfg.learning_rate
                    self.network_optimizer.param_groups[0]["lr"] = curr_lr

                    # Update parameters for policy and value networks
                    self.model.network.to(self._torch_device)
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
                    total_steps += self._cfg.epoch_num_trials * self._cfg.num_rollout_steps
                    num_updates += 1
                    run_session.log_metrics(
                        model_iteration=iteration_info.iteration,
                        policy_loss=policy_loss.item(),
                        value_loss=value_loss.item(),
                        avg_rewards=avg_rewards.item(),
                        num_steps=total_steps,
                        num_updates=num_updates,
                    )
                    if num_updates % self._cfg.logging_interval == 0:
                        log.info(f"Steps: #{total_steps} | Avg. reward: {avg_rewards.item():.2f}")

                    # Publish the newly updated model
                    self.model.iter_idx = iter_idx
                    serialized_model = PPOModel.serialize_model(self.model)

                    if num_updates % 50 == 0:
                        iteration_info = await run_session.model_registry.store_model(
                            name=model_id, model=serialized_model
                        )
                    else:
                        iteration_info = await run_session.model_registry.publish_model(
                            name=model_id, model=serialized_model
                        )

                    self.model.network.to(self._torch_device)
            if total_steps > self._cfg.max_training_steps:
                break
        iteration_info = await run_session.model_registry.store_model(name=model_id, model=serialized_model)
