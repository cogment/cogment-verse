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
from typing import List, Tuple

import cogment
import numpy as np
import torch
from gym.spaces import Discrete, utils
from torch.distributions.distribution import Distribution

from cogment_verse import HumanDataBuffer, Model
from cogment_verse.run.run_session import RunSession
from cogment_verse.run.sample_producer_worker import SampleProducerSession
from cogment_verse.specs import (
    HUMAN_ACTOR_IMPL,
    WEB_ACTOR_NAME,
    ActorClass,
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
        self.actor = initialize_layer(torch.nn.Linear(512, num_actions), 0.01)
        self.value = initialize_layer(torch.nn.Linear(512, 1), 1)

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute the value of being in a state"""
        observation = observation / 255.0
        return self.value(self.shared_network(observation))

    def get_action(self, observation: torch.Tensor) -> Distribution:
        """Actions given observations"""
        observation = observation / 255.0
        action_logits = self.actor(self.shared_network(observation))
        dist = torch.distributions.categorical.Categorical(logits=action_logits)

        return dist


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
                model.network.to(torch.device("cpu")).state_dict(),
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
            device=model_user_data["device"],
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
        return [ActorClass.PLAYER.value]

    async def impl(self, actor_session):
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
        model = await PPOModel.retrieve_model(actor_session.model_registry, config.model_id, config.model_iteration)
        model.network.eval()

        log.info(f"Actor - retreved model number: {model.iteration}")
        obs_shape = model.input_shape[::-1]

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                if (
                    event.observation.observation.HasField("current_player")
                    and event.observation.observation.current_player != actor_session.name
                ):
                    # Not the turn of the agent
                    actor_session.do_action(action_space.serialize(action_space.create()))
                    continue
                obs = observation_space.deserialize(event.observation.observation)
                obs_tensor = torch.tensor(obs.flat_value, dtype=self._dtype).reshape(obs_shape)
                obs_tensor = torch.unsqueeze(obs_tensor.permute((2, 0, 1)), dim=0).to(torch.device(model.device))

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
        "action_loss_coef": 1.0,
        "clipping_coef": 0.1,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "num_steps": -1,  # End of trial
        "lambda_gae": 0.95,
        "device": "cpu",
        "grad_norm": 0.5,
        "lr_decay_factor": 0.999,
        "image_size": [6, 84, 84],
        "buffer_capacity": 100_000,
    }

    def __init__(self, environment_specs: EnvironmentSpecs, cfg: EnvironmentConfig) -> None:
        super().__init__()
        self._dtype = torch.float32
        self._environment_specs = environment_specs
        self._cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.returns = 0

        # Set random seed for initializing neural network parameters
        torch.manual_seed(self._cfg.seed)
        self.model = PPOModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            num_actions=utils.flatdim(self._environment_specs.get_action_space().gym_space),
            input_shape=tuple(self._cfg.image_size),
            num_policy_outputs=1,
            n_iter=self._cfg.num_epochs,
            device=self._cfg.device,
            dtype=self._dtype,
        )

        # Get optimizer for two models
        self.network_optimizer = torch.optim.Adam(self.model.network.parameters(), lr=self._cfg.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.network_optimizer, step_size=1000, gamma=0.1)

    async def trial_sample_sequences_producer_impl(self, sample_producer_session: SampleProducerSession):
        """Collect sample from the trial"""
        observation = []
        action = []
        reward = []
        done = []
        current_players = []

        actor_params = {
            actor_params.name: actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == ActorClass.PLAYER.value
        }
        actor_names = list(actor_params.keys())
        player_environment_specs = EnvironmentSpecs.deserialize(actor_params[actor_names[0]].config.environment_specs)
        player_observation_space = player_environment_specs.get_observation_space()
        player_action_space = player_environment_specs.get_action_space()
        obs_shape = tuple(self._cfg.image_size)[::-1]
        player_actor_name = actor_names[0]

        async for sample in sample_producer_session.all_trial_samples():
            previous_actor_sample = sample.actors_data[player_actor_name]
            player_actor_name = previous_actor_sample.observation.current_player
            actor_sample = sample.actors_data[player_actor_name]
            obs_flat = player_observation_space.deserialize(actor_sample.observation).flat_value
            observation_value = torch.unsqueeze(
                torch.permute(torch.tensor(obs_flat, dtype=self._dtype).reshape(obs_shape), (2, 0, 1)), dim=0
            )
            observation.append(observation_value)
            if sample.trial_state == cogment.TrialState.ENDED:
                done.append(torch.ones(1, dtype=self._dtype))
                break

            action_value = torch.tensor(player_action_space.deserialize(actor_sample.action).value, dtype=self._dtype)
            reward_value = torch.tensor(
                actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype
            )
            action.append(action_value)
            reward.append(reward_value)
            done.append(torch.zeros(1, dtype=self._dtype))
            current_players.append(player_actor_name)

        # Keeping the samples grouped by trial by emitting only one grouped sample at the end of the trial
        sample_producer_session.produce_sample((observation, action, reward, done, current_players))

    @abstractmethod
    async def impl(self, run_session: RunSession) -> dict:
        raise NotImplementedError

    async def train_step(
        self,
        observations: List[torch.Tensor],
        rewards: List[torch.Tensor],
        actions: List[torch.Tensor],
        dones: List[torch.Tensor],
    ):
        """Train the model after collecting the data from the trial"""
        # Take n steps from the rollout
        _observations = torch.vstack(observations).to(self._device)
        _actions = torch.vstack(actions).to(self._device)
        _rewards = torch.vstack(rewards).to(self._device)
        _dones = torch.vstack(dones).to(self._device)

        # Mask to remove values and observations from done samples.
        mask = torch.squeeze(_dones) == 0
        masked_observations = _observations[mask]

        # Make a dataloader in order to process data in batch
        adv_batch_state = self.make_dataloader(_observations, self._cfg.batch_size, self.model.input_shape)
        values = self.compute_value(adv_batch_state)
        next_values = values * (1 - _dones)

        # Compute the generalized advantage estimation
        advs = self.compute_gae(
            rewards=_rewards,
            values=next_values,
            dones=_dones,
            gamma=self._cfg.discount_factor,
            lam=self._cfg.lambda_gae,
        )
        batch_state = self.make_dataloader(masked_observations, self._cfg.batch_size, self.model.input_shape)
        batch_action = self.make_dataloader(_actions, self._cfg.batch_size, (self.model.num_policy_outputs,))
        log_probs = self.compute_batch_log_lik(batch_state, batch_action)

        # Update parameters for policy and value networks
        policy_loss, value_loss = self.update_parameters(
            observations=masked_observations,
            actions=_actions,
            advs=advs,
            values=values[mask],
            log_probs=log_probs,
            num_epochs=self._cfg.num_epochs,
        )
        return policy_loss, value_loss

    def update_parameters(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advs: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        num_epochs: int,
    ) -> Tuple[torch.Tensor]:
        """Update policy & value networks"""

        returns = advs + values
        num_obs = len(returns)
        global_idx = np.arange(num_obs)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        for _ in range(num_epochs):
            np.random.shuffle(global_idx)
            for i in range(0, num_obs, self._cfg.batch_size):
                # Get data in batch. TODO: Send data to device (need to test with cuda)
                # idx = np.random.randint(0, num_obs, self._cfg.batch_size)
                # idx = np.random.choice(num_obs, self._cfg.batch_size, replace=False)
                idx = global_idx[i : i + self._cfg.batch_size]
                if len(idx) < self._cfg.batch_size:
                    break
                observation = observations[idx]
                action = actions[idx]
                return_ = returns[idx]
                adv = advs[idx]
                old_value = values[idx]
                old_log_prob = log_probs[idx]

                # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Compute the value and values loss
                value = self.model.network.get_value(observation)
                # value_loss = torch.nn.functional.mse_loss(return_, value) * self._cfg.value_loss_coef
                value_loss_unclipped = (value - return_) ** 2
                value_clipped = old_value + torch.clamp(value - old_value, -0.1, 0.1)
                value_loss_clipped = (value_clipped - return_) ** 2
                value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
                value_loss = value_loss_max.mean() * self._cfg.value_loss_coef

                # Get action distribution & the log-likelihood
                dist = self.model.network.get_action(observation)
                new_log_prob = dist.log_prob(action.flatten()).view(-1, 1)
                entropy = dist.entropy()
                ratio = torch.exp(new_log_prob - old_log_prob)

                # Compute policy loss
                policy_loss_1 = -adv * ratio
                policy_loss_2 = -adv * torch.clamp(ratio, 1 - self._cfg.clipping_coef, 1 + self._cfg.clipping_coef)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Loss
                entropy_loss = entropy.mean()
                loss = policy_loss - self._cfg.entropy_loss_coef * entropy_loss + value_loss

                # Update value network
                self.network_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), self._cfg.grad_norm)
                self.network_optimizer.step()

        # Decaying learning rate after each update
        self._cfg.learning_rate = max(self._cfg.lr_decay_factor * self._cfg.learning_rate, 0.000001)
        self.network_optimizer.param_groups[0]["lr"] = self._cfg.learning_rate
        self._cfg.clipping_coef = max(self._cfg.lr_decay_factor * self._cfg.clipping_coef, 0.05)
        # log.info(f"learning rate {self.network_optimizer.param_groups[0]['lr']}")
        # self.model.scheduler.step()

        return policy_loss, value_loss

    @staticmethod
    async def compute_average_reward(rewards: list) -> float:
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
        output_batches = torch.zeros((batch_size, *obs_shape), dtype=self._dtype).to(self._device)
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
                output_batches = torch.zeros((batch_size, *obs_shape), dtype=self._dtype).to(self._device)
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
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float = 0.99, lam: float = 0.95
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation. See equations 11 & 12 in
        https://arxiv.org/pdf/1707.06347.pdf
        """

        advs = []
        with torch.no_grad():
            gae = 0.0
            # dones = torch.cat((dones, torch.zeros(1, 1).to(self._device)), dim=0)
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i + 1]) - values[i]
                gae = delta + gamma * lam * (1 - dones[i + 1]) * gae
                advs.append(gae)
        advs.reverse()
        return torch.vstack(advs)


class PPOSelfTraining(BasePPOTraining):
    """Train PPO agent"""

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish the model"""
        model_id = f"{run_session.run_id}_model"

        # Initalize model
        self.model.model_id = model_id  # pylint: disable=attribute-defined-outside-init
        serialized_model = PPOModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )
        self.model.network.to(self._device)

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
            hill_type="none",
        )

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx: int, iter_idx: int):
            agent_actor_params = cogment.ActorParameters(
                cog_settings,
                name="player",
                class_name=ActorClass.PLAYER.value,
                implementation="actors.ppo_atari_pz.PPOActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                    model_id=model_id,
                    model_iteration=iteration_info.iteration,
                    seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
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
        episode_lens = []

        num_updates = 0
        total_steps = 0
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
                (trial_observation, trial_action, trial_reward, trial_done, _) = sample

                observations.extend(trial_observation)
                actions.extend(trial_action)
                rewards.extend(trial_reward)
                dones.extend(trial_done)

                episode_rewards.append(torch.vstack(trial_reward).sum())
                episode_lens.append(torch.tensor(len(trial_action), dtype=torch.float32))
                if len(actions) >= self._cfg.num_steps * self._cfg.epoch_num_trials + 1:
                    num_updates += 1
                    total_steps += self._cfg.num_steps * self._cfg.epoch_num_trials
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
                        avg_lens = await self.compute_average_reward(episode_lens) / self._environment_specs.num_players
                        log.info(f"epoch #{iter_idx + 1}/{self._cfg.num_iter}| avg. len: {avg_lens:0.2f}]")

                        run_session.log_metrics(
                            model_iteration=iteration_info.iteration,
                            policy_loss=policy_loss.item(),
                            value_loss=value_loss.item(),
                            avg_rewards=avg_rewards.item(),
                            avg_lens=avg_lens.item(),
                            num_steps=total_steps,
                            num_updates=num_updates,
                        )

                    # Publish the newly updated model
                    self.model.iter_idx = iter_idx
                    serialized_model = PPOModel.serialize_model(self.model)

                    if num_updates % 50 == 0:
                        iteration_info = await run_session.model_registry.store_model(
                            name=model_id,
                            model=serialized_model,
                        )
                    else:
                        iteration_info = await run_session.model_registry.publish_model(
                            name=model_id,
                            model=serialized_model,
                        )

                    self.model.network.to(self._device)


class HillPPOTraining(BasePPOTraining):
    """Train PPO agent using human's actions"""

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish the model"""
        model_id = f"{run_session.run_id}_model"

        # Initalize model
        self.model.model_id = model_id  # pylint: disable=attribute-defined-outside-init
        serialized_model = PPOModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )
        self.model.network.to(self._device)

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
            capacity=self._cfg.buffer_capacity,
            action_dtype=np.int32,
            file_name=f"{human_data_category}_{run_session.run_id}",
            seed=self._cfg.seed,
        )

        # Create actor parameters
        def create_actor_params(
            actor_names: List[str],
            trial_idx: int,
            iter_idx: int,
            hill_training_trial_period: int,
            iteration: int = -1,
        ):
            np.random.default_rng(self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials)
            human_actor_idx = np.random.choice(len(actor_names), 1, replace=False)
            human = True
            actors = []
            for i, name in enumerate(actor_names):
                if human and i == human_actor_idx:
                    actor = cogment.ActorParameters(
                        cog_settings,
                        name=WEB_ACTOR_NAME,
                        class_name=ActorClass.PLAYER.value,
                        implementation=HUMAN_ACTOR_IMPL,
                        config=AgentConfig(
                            run_id=run_session.run_id,
                            environment_specs=self._environment_specs.serialize(),
                            seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
                        ),
                    )
                else:
                    actor = cogment.ActorParameters(
                        cog_settings,
                        name=name,
                        class_name=ActorClass.PLAYER.value,
                        implementation="actors.ppo_atari_pz.PPOActor",
                        config=AgentConfig(
                            run_id=run_session.run_id,
                            environment_specs=self._environment_specs.serialize(),
                            model_id=model_id,
                            model_iteration=iteration,
                            seed=self._cfg.seed + trial_idx + iter_idx * self._cfg.epoch_num_trials,
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
        observations = []
        actions = []
        rewards = []
        dones = []
        episode_rewards = []
        episode_lens = []

        num_updates = 0
        total_steps = 0
        for iter_idx in range(self._cfg.num_iter):
            # TODO: actor names should not be hard-coding
            for (_, _, _, sample) in run_session.start_and_await_trials(
                trials_id_and_params=[
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
                ],
                sample_producer_impl=self.trial_sample_sequences_producer_impl,
                num_parallel_trials=self._cfg.num_parallel_trials,
            ):
                # Collect the rollout
                (trial_observation, trial_action, trial_reward, trial_done, trial_actor) = sample
                trial_human_observation = [
                    obs for (obs, actor_name) in zip(trial_observation, trial_actor) if actor_name == WEB_ACTOR_NAME
                ]
                trial_human_action = [
                    act for (act, actor_name) in zip(trial_action, trial_actor) if actor_name == WEB_ACTOR_NAME
                ]
                for (obs, act) in zip(trial_human_observation, trial_human_action):
                    human_data_buffer.add(observation=obs, action=act)

                observations.extend(trial_observation)
                actions.extend(trial_action)
                rewards.extend(trial_reward)
                dones.extend(trial_done)

                episode_rewards.append(torch.vstack(trial_reward).sum())
                episode_lens.append(torch.tensor(len(trial_action), dtype=torch.float32))
                if len(actions) >= self._cfg.num_steps * self._cfg.epoch_num_trials + 1:
                    num_updates += 1
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
                        avg_lens = await self.compute_average_reward(episode_lens) / self._environment_specs.num_players
                        log.info(f"epoch #{iter_idx + 1}/{self._cfg.num_iter}| avg. len: {avg_lens:0.2f}")

                        run_session.log_metrics(
                            model_iteration=iteration_info.iteration,
                            policy_loss=policy_loss.item(),
                            value_loss=value_loss.item(),
                            avg_rewards=avg_rewards.item(),
                            avg_lens=avg_lens.item(),
                            num_steps=total_steps,
                            num_updates=num_updates,
                        )

                    # Publish the newly updated model
                    self.model.iter_idx = iter_idx
                    serialized_model = PPOModel.serialize_model(self.model)

                    if num_updates % 50 == 0:
                        iteration_info = await run_session.model_registry.store_model(
                            name=model_id,
                            model=serialized_model,
                        )
                    else:
                        iteration_info = await run_session.model_registry.publish_model(
                            name=model_id,
                            model=serialized_model,
                        )
                    self.model.network.to(self._device)


class HumanFeedbackPPOTraining(BasePPOTraining):
    """Train PPO agent with human feedback"""

    async def trial_sample_sequences_producer_impl(self, sample_producer_session: SampleProducerSession):
        """Collect sample from the trial"""
        observation = []
        action = []
        reward = []
        done = []
        human_observation = []
        human_reward = []

        actor_params = {
            actor_params.name: actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == ActorClass.PLAYER.value
        }
        actor_names = list(actor_params.keys())
        player_environment_specs = EnvironmentSpecs.deserialize(actor_params[actor_names[0]].config.environment_specs)
        player_observation_space = player_environment_specs.get_observation_space()
        player_action_space = player_environment_specs.get_action_space()
        obs_shape = tuple(self._cfg.image_size)[::-1]
        player_actor_name = actor_names[0]

        async for sample in sample_producer_session.all_trial_samples():
            if sample.trial_state == cogment.TrialState.ENDED:
                obs_flat = player_observation_space.deserialize(actor_sample.observation).flat_value
                observation_value = torch.unsqueeze(
                    torch.permute(torch.tensor(obs_flat, dtype=self._dtype).reshape(obs_shape), (2, 0, 1)), dim=0
                )
                observation.append(observation_value)
                done.append(torch.ones(1, dtype=self._dtype))
                break
            previous_actor_sample = sample.actors_data[player_actor_name]
            player_actor_name = previous_actor_sample.observation.current_player
            actor_sample = sample.actors_data[player_actor_name]
            if player_actor_name != WEB_ACTOR_NAME:
                obs_flat = player_observation_space.deserialize(actor_sample.observation).flat_value
                observation_value = torch.unsqueeze(
                    torch.permute(torch.tensor(obs_flat, dtype=self._dtype).reshape(obs_shape), (2, 0, 1)), dim=0
                )
                action_value = torch.tensor(
                    player_action_space.deserialize(actor_sample.action).value, dtype=self._dtype
                )

                reward_value = torch.tensor(
                    actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype
                )
                observation.append(observation_value)
                action.append(action_value)
                reward.append(reward_value)
                done.append(torch.zeros(1, dtype=self._dtype))
            else:
                obs_flat = player_observation_space.deserialize(actor_sample.observation).flat_value
                observation_value = torch.unsqueeze(
                    torch.permute(torch.tensor(obs_flat, dtype=self._dtype).reshape(obs_shape), (2, 0, 1)), dim=0
                )
                reward_value = torch.tensor(
                    actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype
                )
                human_observation.append(observation_value)
                human_reward.append(reward_value)

        # Keeping the samples grouped by trial by emitting only one grouped sample at the end of the trial
        sample_producer_session.produce_sample((observation, action, reward, done, human_reward))

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish the model"""
        model_id = f"{run_session.run_id}_model"

        # Initalize model
        self.model.model_id = model_id  # pylint: disable=attribute-defined-outside-init
        serialized_model = PPOModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )
        self.model.network.to(self._device)

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
            hill_type="feedback",
        )

        # Human data buffer
        human_data_category = "feedback"
        human_data_buffer = HumanDataBuffer(
            observation_shape=tuple(self._cfg.image_size),
            action_shape=(1,),
            human_data_category=human_data_category,
            capacity=self._cfg.buffer_capacity,
            action_dtype=np.int32,
            file_name=f"{human_data_category}_{run_session.run_id}",
            seed=self._cfg.seed,
        )

        # Create actor parameters
        def create_actor_params(
            actor_names: List[str], trial_idx: int, hill_training_trial_period: int, iteration: int = -1
        ):
            actors = []
            for i, name in enumerate(actor_names):
                actor = cogment.ActorParameters(
                    cog_settings,
                    name=name,
                    class_name=ActorClass.PLAYER.value,
                    implementation="actors.ppo_atari_pz.PPOActor",
                    config=AgentConfig(
                        run_id=run_session.run_id,
                        environment_specs=self._environment_specs.serialize(),
                        model_id=model_id,
                        model_iteration=iteration,
                        seed=self._cfg.seed,
                    ),
                )
                actors.append(actor)

            actor = cogment.ActorParameters(
                cog_settings,
                name=WEB_ACTOR_NAME,
                class_name=ActorClass.EVALUATOR.value,
                implementation=HUMAN_ACTOR_IMPL,
                config=AgentConfig(run_id=run_session.run_id, environment_specs=self._environment_specs.serialize()),
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
        observations = []
        actions = []
        rewards = []
        dones = []
        human_rewards = []
        episode_rewards = []
        episode_lens = []

        num_updates = 0
        total_steps = 0
        for iter_idx in range(self._cfg.num_iter):
            # TODO: actor names should not be hard-coding
            for (_, _, _, sample) in run_session.start_and_await_trials(
                trials_id_and_params=[
                    (
                        f"{run_session.run_id}_{iter_idx}_{trial_idx}",
                        create_trial_params(
                            trial_idx,
                            iter_idx,
                            actors=create_actor_params(
                                actor_names=["first_0", "second_0"],
                                trial_idx=trial_idx,
                                hill_training_trial_period=hill_training_trial_period,
                            ),
                        ),
                    )
                    for trial_idx in range(self._cfg.epoch_num_trials)
                ],
                sample_producer_impl=self.trial_sample_sequences_producer_impl,
                num_parallel_trials=self._cfg.num_parallel_trials,
            ):
                # Collect the rollout
                (trial_observation, trial_action, trial_reward, trial_done, trial_human_reward) = sample

                for (obs, act, rew) in zip(trial_observation, trial_action, trial_human_reward):
                    human_data_buffer.add(observation=obs, action=act, feedback=rew)

                observations.extend(trial_observation)
                actions.extend(trial_action)
                rewards.extend(trial_reward)
                dones.extend(trial_done)
                human_rewards.extend(trial_human_reward)

                episode_rewards.append(torch.tensor(len(trial_action), dtype=torch.float32))
                episode_lens.append(torch.tensor(len(trial_action), dtype=torch.float32))
                if len(actions) >= self._cfg.num_steps * self._cfg.epoch_num_trials + 1:
                    num_updates += 1
                    # Update model parameters
                    train_rewards = [rew + human_rew for (rew, human_rew) in zip(rewards, human_rewards)]
                    policy_loss, value_loss = await self.train_step(
                        observations=observations, rewards=train_rewards, actions=actions, dones=dones
                    )

                    # Reset the data storage
                    observations = []
                    actions = []
                    rewards = []
                    dones = []
                    if iter_idx % 100 == 0:
                        # Compute average rewards for last 100 episodes
                        avg_rewards = await self.compute_average_reward(episode_rewards)
                        avg_lens = await self.compute_average_reward(episode_lens) / self._environment_specs.num_players
                        log.info(f"epoch #{iter_idx + 1}/{self._cfg.num_iter}| avg. len: {avg_lens:0.2f}")

                        run_session.log_metrics(
                            model_iteration=iteration_info.iteration,
                            policy_loss=policy_loss.item(),
                            value_loss=value_loss.item(),
                            avg_rewards=avg_rewards.item(),
                            avg_lens=avg_lens.item(),
                            num_steps=total_steps,
                            num_updates=num_updates,
                        )

                    # Publish the newly updated model
                    self.model.iter_idx = iter_idx
                    serialized_model = PPOModel.serialize_model(self.model)

                    if num_updates % 50 == 0:
                        iteration_info = await run_session.model_registry.store_model(
                            name=model_id,
                            model=serialized_model,
                        )
                    else:
                        iteration_info = await run_session.model_registry.publish_model(
                            name=model_id,
                            model=serialized_model,
                        )
                    self.model.network.to(self._device)
