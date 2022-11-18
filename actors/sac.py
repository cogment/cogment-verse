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
import time
from typing import Tuple

import cogment
import numpy as np
import torch
from torch.distributions.normal import Normal

from cogment_verse import Model, TorchReplayBuffer, TorchReplayBufferSample
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
    get_action_bounds,
    unflatten,
)

# torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)

log = logging.getLogger(__name__)
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# pylint: disable=E1102
# pylint: disable=W0212
class PolicyNetwork(torch.nn.Module):
    """Gaussian policy network where action is modeled by Gaussian distribution"""

    def __init__(self, num_input: int, num_output: int, num_hidden: int) -> None:
        super().__init__()
        self.input = torch.nn.Linear(num_input, num_hidden)
        self.fully_connected = torch.nn.Linear(num_hidden, num_hidden)
        self.mean = torch.nn.Linear(num_hidden, num_output)
        self.log_std = torch.nn.Linear(num_hidden, num_output)
        # self.log_std = torch.nn.Parameter(torch.zeros(1, num_output))

    def forward(self, x: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        # Input layer
        x = self.input(x)
        x = torch.nn.functional.relu(x)

        # Hidden layer
        x = self.fully_connected(x)
        x = torch.nn.functional.relu(x)

        # Output layer
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std


class ValueNetwork(torch.nn.Module):
    """Value network that quantifies the quality of an action given a state."""

    def __init__(self, num_input: int, num_hidden: int):
        super().__init__()
        # Value network
        self.input = torch.nn.Linear(num_input, num_hidden)
        self.fully_connected = torch.nn.Linear(num_hidden, num_hidden)
        self.output = torch.nn.Linear(num_hidden, 1)

    def forward(self, observation_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Value network
        x_v = self.input(observation_action)
        x_v = torch.nn.functional.relu(x_v)
        x_v = self.fully_connected(x_v)
        x_v = torch.nn.functional.relu(x_v)
        value = self.output(x_v)
        return value


def initialize_weight(param) -> None:
    """Orthogonal initialization of the weight's values of a network"""
    if isinstance(param, torch.nn.Linear):
        torch.nn.init.orthogonal_(param.weight.data)
        torch.nn.init.constant_(param.bias.data, 0)


class SACModel(Model):
    """Soft-Actor Critic (SAC) https://arxiv.org/abs/1801.01290"""

    target_entropy: torch.Tensor
    log_alpha: torch.Tensor
    # alpha_optimizer: torch.optim.Optimizer

    def __init__(
        self,
        model_id: int,
        environment_implementation: str,
        num_inputs: int,
        num_outputs: int,
        policy_network_hidden_nodes: int,
        value_network_hidden_nodes: int,
        alpha: float,
        learning_rate: float = 0.01,
        dtype: torch.FloatTensor = torch.float32,
        version_number: int = 0,
        is_alpha_learnable: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(model_id, version_number)
        self.model_id = model_id
        self.environment_implementation = environment_implementation
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.policy_network_hidden_nodes = policy_network_hidden_nodes
        self.value_network_hidden_nodes = value_network_hidden_nodes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.dtype = dtype
        self.version_number = version_number
        self.device = device

        # Networks
        self.policy_network = PolicyNetwork(
            num_input=self.num_inputs, num_hidden=self.policy_network_hidden_nodes, num_output=self.num_outputs
        ).to(self.device)
        self.value_network_1 = ValueNetwork(
            num_input=self.num_inputs + self.num_outputs, num_hidden=self.value_network_hidden_nodes
        ).to(self.device)
        self.value_network_2 = ValueNetwork(
            num_input=self.num_inputs + self.num_outputs, num_hidden=self.value_network_hidden_nodes
        ).to(self.device)
        self.target_network_1 = ValueNetwork(
            num_input=self.num_inputs + self.num_outputs, num_hidden=self.value_network_hidden_nodes
        ).to(self.device)
        self.target_network_2 = ValueNetwork(
            num_input=self.num_inputs + self.num_outputs, num_hidden=self.value_network_hidden_nodes
        ).to(self.device)

        # Intialize networks's parameters
        self.policy_network.apply(initialize_weight)
        self.value_network_1.apply(initialize_weight)
        self.value_network_2.apply(initialize_weight)
        self.target_network_1.load_state_dict(self.value_network_1.state_dict())
        self.target_network_2.load_state_dict(self.value_network_2.state_dict())

        # Get optimizer for two models
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer_1 = torch.optim.Adam(self.value_network_1.parameters(), lr=learning_rate)
        self.value_optimizer_2 = torch.optim.Adam(self.value_network_2.parameters(), lr=learning_rate)

        # Learnable alpha
        if is_alpha_learnable:
            self.target_entropy = -torch.tensor(float(self.num_outputs)).to(self.device).item()
            self.log_alpha = (torch.zeros(1, dtype=self.dtype, device=self.device)).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.log_alpha = torch.log(torch.ones(1, dtype=self.dtype, device=self.device) * self.alpha)

        # Learning schedule
        self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.99)
        self.value_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.value_optimizer_1, gamma=0.99)

        # version user data
        self.iter_idx = 0
        self.total_samples = 0

    @property
    def device(self):
        """Get device"""
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        """Set device"""
        if value == "cuda" and torch.cuda.is_available():
            self._device = torch.device(value)
        else:
            self._device = torch.device("cpu")

    def get_model_user_data(self) -> dict:
        """Get user model"""
        return {
            "environment_implementation": self.environment_implementation,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "policy_network_hidden_nodes": self.policy_network_hidden_nodes,
            "value_network_hidden_nodes": self.value_network_hidden_nodes,
            "alpha": self.alpha,
        }

    def save(self, model_data_f: str) -> dict:
        """Save the model"""
        torch.save(
            (self.policy_network.state_dict(), self.value_network_1.state_dict(), self.value_network_2.state_dict()),
            model_data_f,
        )
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
            alpha=model_user_data["alpha"],
        )

        # Load the model parameters
        (policy_network_state_dict, value_network_1_state_dict, value_network_2_state_dict) = torch.load(model_data_f)
        model.policy_network.load_state_dict(policy_network_state_dict)
        model.value_network_1.load_state_dict(value_network_1_state_dict)
        model.value_network_2.load_state_dict(value_network_2_state_dict)

        # Load version data
        model.iter_idx = version_user_data["iter_idx"]
        model.total_samples = version_user_data["total_samples"]
        return model

    def policy_sampler(
        self, observation: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, reparam: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log-likelihood"""
        mean, log_std = self.policy_network(observation)

        # Ensure the log of standard deviation are not exposed during trainign
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.normal.Normal(mean, std)

        # Reparametrization trick i.e., action = mu + std * N(0, 1)
        if reparam:
            action_sample = dist.rsample()
        else:
            action_sample = dist.sample()

        # Transform to tanh space
        action_transform = torch.tanh(action_sample)
        mean_transform = torch.tanh(mean) * scale + bias

        # Rescale
        action = action_transform * scale + bias

        # Log-likelihood in transform space
        log_prob = dist.log_prob(action_sample)
        log_prob_transform = torch.log(scale * (1.0 - (action_transform**2)) + 1e-6)
        log_prob -= log_prob_transform
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean_transform


class SACActor:
    """Soft actor critic actor"""

    def __init__(self, _cfg):
        self._dtype = torch.float
        self._cfg = _cfg

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
        action_min, action_max = get_action_bounds(config.environment_specs.action_space)
        scale = torch.tensor((action_max - action_min) / 2.0, dtype=self._dtype)
        bias = torch.tensor((action_max + action_min) / 2.0, dtype=self._dtype)

        # Get observation and action space
        observation_space = config.environment_specs.observation_space
        action_space = config.environment_specs.action_space

        # Get model
        model, _, _ = await actor_session.model_registry.retrieve_version(SACModel, config.model_id, -1)
        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                obs_tensor = torch.tensor(
                    flatten(observation_space, event.observation.observation.value), dtype=self._dtype
                ).view(1, -1)

                # Get action from policy network
                with torch.no_grad():
                    action, _, _ = model.policy_sampler(observation=obs_tensor, scale=scale, bias=bias)
                    action = action.cpu().numpy()[0]

                # Send action to environment
                action_value = unflatten(action_space, action)
                actor_session.do_action(PlayerAction(value=action_value))


class SACTraining:
    """Train SAC agent"""

    default_cfg = {
        "seed": 10,
        "num_trials": 5000,
        "num_parallel_trials": 1,
        "discount_factor": 0.99,
        "entropy_loss_coef": 0.05,
        "value_loss_coef": 0.5,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "grad_norm": 0.5,
        "is_alpha_learnable": False,
        "alpha": 0.2,
        "tau": 0.005,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "device": "cpu",
        "delay_steps": 2,
        "policy_network": {"num_hidden_nodes": 256},
        "value_network": {"num_hidden_nodes": 256},
    }
    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(self, environment_specs: EnvironmentSpecs, cfg: EnvironmentConfig) -> None:
        super().__init__()
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg
        self._device = torch.device(self._cfg.device)
        self._rng = np.random.default_rng(self._cfg.seed)
        self.returns = 0
        action_min, action_max = get_action_bounds(self._environment_specs.action_space)
        self.action_scale = torch.tensor((action_max - action_min) / 2.0, dtype=self._dtype)
        self.action_bias = torch.tensor((action_max + action_min) / 2.0, dtype=self._dtype)

        self.model = SACModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            num_inputs=flattened_dimensions(self._environment_specs.observation_space),
            num_outputs=flattened_dimensions(self._environment_specs.action_space),
            learning_rate=self._cfg.learning_rate,
            policy_network_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
            value_network_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
            alpha=self._cfg.alpha,
            is_alpha_learnable=self._cfg.is_alpha_learnable,
            dtype=self._dtype,
        )

    async def sample_producer_impl(self, sample_producer_session: SampleProducerSession):
        """Collect sample from the trial"""
        observation = []
        action = []
        reward = []
        done = []

        player_actor_params = sample_producer_session.trial_info.parameters.actors[0]
        player_actor_name = player_actor_params.name
        player_observation_space = player_actor_params.config.environment_specs.observation_space
        player_action_space = player_actor_params.config.environment_specs.action_space

        observation = None
        action = None
        reward = None
        total_reward = 0
        async for sample in sample_producer_session.all_trial_samples():
            actor_sample = sample.actors_data[player_actor_name]
            if actor_sample.observation is None:
                # This can happen when there is several "end-of-trial" samples
                continue
            next_observation = torch.tensor(
                flatten(player_observation_space, actor_sample.observation.value), dtype=self._dtype
            )
            if observation is not None:
                done = sample.trial_state == cogment.TrialState.ENDED
                sample_producer_session.produce_sample(
                    (
                        observation,
                        next_observation,
                        action,
                        reward,
                        torch.ones(1, dtype=torch.float32) if done else torch.zeros(1, dtype=torch.float32),
                        total_reward,
                    )
                )
                if done:
                    break

            observation = next_observation
            action = torch.tensor(flatten(player_action_space, actor_sample.action.value), dtype=self._dtype)
            reward = torch.tensor([actor_sample.reward] if actor_sample.reward is not None else 0, dtype=self._dtype)
            total_reward += reward.item()

    async def impl(self, run_session: RunSession) -> dict:
        """Train and publish the model"""
        # Initializing a model
        model_id = f"{run_session.run_id}_model"
        assert self._environment_specs.num_players == 1
        assert len(self._environment_specs.action_space.properties) == 1
        assert self._environment_specs.action_space.properties[0].WhichOneof("type") == "box"

        self.model.model_id = model_id
        _, version_info = await run_session.model_registry.publish_initial_version(self.model)

        run_session.log_params(
            self._cfg, model_id=model_id, environment_implementation=self._environment_specs.implementation
        )

        # Initalizing replay buffer
        replay_buffer = TorchReplayBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=(flattened_dimensions(self._environment_specs.observation_space),),
            observation_dtype=self._dtype,
            action_shape=(flattened_dimensions(self._environment_specs.action_space),),
            action_dtype=self._dtype,
            reward_dtype=self._dtype,
            seed=self._cfg.seed,
        )

        start_time = time.time()
        total_reward_acc = 0

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx: int):
            agent_actor_params = cogment.ActorParameters(
                cog_settings,
                name="player",
                class_name=PLAYER_ACTOR_CLASS,
                implementation="actors.sac.SACActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs,
                    model_id=model_id,
                    model_version=-1,
                ),
            )

            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id,
                    render=False,
                    seed=self._cfg.seed + trial_idx,
                ),
                actors=[agent_actor_params],
            )

        # Run environment
        for (step_idx, _, trial_idx, sample,) in run_session.start_and_await_trials(
            trials_id_and_params=[
                (f"{run_session.run_id}_{trial_idx}", create_trial_params(trial_idx))
                for trial_idx in range(self._cfg.num_trials)
            ],
            sample_producer_impl=self.sample_producer_impl,
            num_parallel_trials=self._cfg.num_parallel_trials,
        ):
            # Collect the rollout
            (observation, next_observation, action, reward, done, total_reward) = sample
            replay_buffer.add(
                observation=observation, next_observation=next_observation, action=action, reward=reward, done=done
            )
            trial_done = done.item() == 1
            if trial_done:
                run_session.log_metrics(trial_idx=trial_idx, total_reward=total_reward)
                total_reward_acc += total_reward
                if (trial_idx + 1) % 100 == 0:
                    total_reward_avg = total_reward_acc / 100
                    run_session.log_metrics(total_reward_avg=total_reward_avg)
                    total_reward_acc = 0
                    log.info(
                        f"[SAC/{run_session.run_id}] trial #{trial_idx + 1}/{self._cfg.num_trials}| steps #{step_idx} | Avg reward: {total_reward_avg:.2f}"
                    )

            # Training steps
            if step_idx > self._cfg.learning_starts and replay_buffer.size() > self._cfg.batch_size:
                data = replay_buffer.sample(self._cfg.batch_size)
                self.model.iter_idx = step_idx
                policy_loss, value_loss, log_alpha = await self.train_step(data=data, num_steps=step_idx)

                # Publish model
                version_info = await run_session.model_registry.publish_version(self.model, archived=False)
                if step_idx % 100 == 0:
                    end_time = time.time()
                    steps_per_seconds = 100 / (end_time - start_time)
                    start_time = end_time
                    run_session.log_metrics(
                        model_version_number=version_info["version_number"],
                        value_loss=value_loss,
                        log_alpha=log_alpha,
                        policy_loss=policy_loss,
                        steps_per_seconds=steps_per_seconds,
                    )
        version_info = await run_session.model_registry.publish_version(self.model, archived=True)

    async def train_step(self, data: TorchReplayBufferSample, num_steps: int) -> Tuple[float, float]:
        """Train the model after collecting the data from the trial"""

        alpha = torch.exp(self.model.log_alpha).item()
        with torch.no_grad():
            next_action, next_action_log_prob, _ = self.model.policy_sampler(
                observation=data.next_observation, scale=self.action_scale, bias=self.action_bias, reparam=True
            )
            observation_action = torch.cat([data.next_observation, next_action], 1)
            next_target_1 = self.model.target_network_1(observation_action)
            next_target_2 = self.model.target_network_2(observation_action)
            min_value = torch.min(next_target_1, next_target_2) - alpha * next_action_log_prob
            return_ = (
                data.reward.reshape(-1, self._cfg.batch_size).T
                + (1 - data.done.reshape(-1, self._cfg.batch_size).T) * self._cfg.discount_factor * min_value
            )

        # Compute loss for value network
        value_1 = self.model.value_network_1(torch.cat([data.observation, data.action], 1))
        value_2 = self.model.value_network_2(torch.cat([data.observation, data.action], 1))
        value_loss_1 = torch.nn.functional.mse_loss(value_1, return_) * self._cfg.value_loss_coef
        value_loss_2 = torch.nn.functional.mse_loss(value_2, return_) * self._cfg.value_loss_coef
        value_loss = value_loss_1 + value_loss_2

        # Update value network
        self.model.value_optimizer_1.zero_grad()
        value_loss_1.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.value_network_1.parameters(), self._cfg.grad_norm)
        self.model.value_optimizer_1.step()

        self.model.value_optimizer_2.zero_grad()
        value_loss_2.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.value_network_2.parameters(), self._cfg.grad_norm)
        self.model.value_optimizer_2.step()

        # Compute loss for policy network
        policy_loss = value_loss_1 * 0
        log_alpha_clone = self.model.log_alpha.clone()
        if num_steps % self._cfg.delay_steps == 0:
            for _ in range(self._cfg.delay_steps):
                action_reparam, action_log_prob_reparam, _ = self.model.policy_sampler(
                    observation=data.observation, scale=self.action_scale, bias=self.action_bias, reparam=True
                )
                value_reparam_1 = self.model.value_network_1(torch.cat([data.observation, action_reparam], 1))
                value_reparam_2 = self.model.value_network_2(torch.cat([data.observation, action_reparam], 1))
                min_value_reparam = torch.min(value_reparam_1, value_reparam_2)
                policy_loss = ((action_log_prob_reparam * alpha) - min_value_reparam).mean()

                # Update policy network
                self.model.policy_optimizer.zero_grad()
                policy_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.policy_network.parameters(), self._cfg.grad_norm)
                self.model.policy_optimizer.step()

            # Update alpha
            if self._cfg.is_alpha_learnable:
                alpha_loss = (
                    -self.model.log_alpha * (action_log_prob_reparam + self.model.target_entropy).detach()
                ).mean()
                self.model.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.model.alpha_optimizer.step()
                log_alpha_clone = self.model.log_alpha.clone()
            else:
                alpha_loss = torch.tensor(0.0).to(self.model.device)
                log_alpha_clone = self.model.log_alpha.clone()

            self.soft_update()

        return value_loss.item(), policy_loss.item(), log_alpha_clone.item()

    def hard_update(self) -> None:
        """Perform a hard update for the target networks where their parameters
        are set to the value network's updated parameters"""
        for target_param, value_param in zip(
            self.model.target_network_1.parameters(), self.model.value_network_1.parameters()
        ):
            target_param.data.copy_(value_param.data)
        for target_param, value_param in zip(
            self.model.target_network_2.parameters(), self.model.value_network_2.parameters()
        ):
            target_param.data.copy_(value_param.data)

    def soft_update(self) -> None:
        """Perform a soft update for the target networks where
        target net's parameters = \tau * previvous target net's parameters + (1 - tau) * value net's parameters
        """
        for target_param, value_param in zip(
            self.model.target_network_1.parameters(), self.model.value_network_1.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - self._cfg.tau) + value_param.data * self._cfg.tau)
        for target_param, value_param in zip(
            self.model.target_network_2.parameters(), self.model.value_network_2.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - self._cfg.tau) + value_param.data * self._cfg.tau)
