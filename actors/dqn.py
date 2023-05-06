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

# pylint: disable=E0611

import logging
import copy
import time
import json
from typing import List, Tuple, Union
import numpy as np

import cogment
import torch
from torch import nn
from gym.spaces import Discrete, utils

from cogment_verse.specs import AgentConfig, cog_settings, EnvironmentConfig, EnvironmentSpecs

from cogment_verse.constants import PLAYER_ACTOR_CLASS

from cogment_verse import Model, TorchReplayBuffer  # pylint: disable=abstract-class-instantiated

torch.manual_seed(0)
np.random.seed(0)
torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)


def create_linear_schedule(start, end, duration):
    slope = (end - start) / duration

    def compute_value(t):
        return max(slope * t + start, end)

    return compute_value


# Acknowledgements: The networks and associated utils are adapted from RLHive


def calculate_output_dim(net, input_shape):
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    placeholder = torch.zeros((0,) + tuple(input_shape))
    output = net(placeholder)
    return output.size()[1:]


class MLPNetwork(nn.Module):
    def __init__(
        self,
        in_dim: Tuple[int],
        hidden_units: Union[int, List[int]] = 256,
    ):
        super().__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        modules = [nn.Linear(np.prod(in_dim), hidden_units[0]), torch.nn.ReLU()]
        for i in range(len(hidden_units) - 1):
            modules.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            modules.append(torch.nn.ReLU())
        self.network = torch.nn.Sequential(*modules)

    def forward(self, x):
        x = x.float()
        if len(x.shape) > 1:
            x = torch.flatten(x, start_dim=1)
        return self.network(x)


class ConvNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        channels=None,
        mlp_layers=None,
        kernel_sizes=1,
        strides=1,
        paddings=0,
        normalization_factor=255,
    ):
        super().__init__()
        self._normalization_factor = normalization_factor
        if channels is not None:
            if isinstance(kernel_sizes, int):
                kernel_sizes = [kernel_sizes] * len(channels)
            if isinstance(strides, int):
                strides = [strides] * len(channels)
            if isinstance(paddings, int):
                paddings = [paddings] * len(channels)

            if not all(len(x) == len(channels) for x in [kernel_sizes, strides, paddings]):
                raise ValueError("The lengths of the parameter lists must be the same")

            # Convolutional Layers
            channels.insert(0, in_dim[0])
            conv_seq = []
            for i in range(0, len(channels) - 1):
                conv_seq.append(
                    torch.nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i],
                    )
                )
                conv_seq.append(torch.nn.ReLU())
            self.conv = torch.nn.Sequential(*conv_seq)
        else:
            self.conv = torch.nn.Identity()

        if mlp_layers is not None:
            # MLP Layers
            conv_output_size = calculate_output_dim(self.conv, in_dim)
            self.mlp = MLPNetwork(conv_output_size, mlp_layers)
        else:
            self.mlp = torch.nn.Identity()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 5:
            x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        x = x.float()
        x = x / self._normalization_factor
        x = self.conv(x)
        x = self.mlp(x)
        return x


class DQNNetwork(nn.Module):
    def __init__(
        self,
        base_network: nn.Module,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.base_network = base_network
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.base_network(x)
        if len(x.shape) > 1:
            x = x.flatten(start_dim=1)
        return self.output_layer(x)


class DQNModel(Model):
    def __init__(
        self,
        model_id,
        environment_implementation,
        num_input,
        num_output,
        num_hidden_nodes,
        epsilon,
        dtype=torch.float,
        version_number=0,
    ):
        super().__init__(model_id, version_number)
        self._dtype = dtype
        self._environment_implementation = environment_implementation
        self._num_input = num_input
        self._num_output = num_output
        self._num_hidden_nodes = list(num_hidden_nodes)

        self.epsilon = epsilon
        self.base_network = MLPNetwork(num_input, self._num_hidden_nodes)
        self.network = DQNNetwork(self.base_network, self._num_hidden_nodes[-1], self._num_output)

        # version user data
        self.num_samples_seen = 0

    def get_model_user_data(self):
        return {
            "environment_implementation": self._environment_implementation,
            "num_input": self._num_input,
            "num_output": self._num_output,
            "num_hidden_nodes": json.dumps(self._num_hidden_nodes),
            "num_samples_seen": self.num_samples_seen,
        }

    @staticmethod
    def serialize_model(model) -> bytes:
        stream = io.BytesIO()
        torch.save(
            (
                model.network.state_dict(),
                model.epsilon,
                model.get_model_user_data(),
            ),
            stream,
        )
        return stream.getvalue()

    @classmethod
    def deserialize_model(cls, serialized_model) -> DQNModel:
        stream = io.BytesIO(serialized_model)
        (network_state_dict, epsilon, model_user_data) = torch.load(stream)

        model = cls(
            model_id=model_user_data["model_id"],
            environment_implementation=model_user_data["environment_implementation"],
            num_input=int(model_user_data["num_input"]),
            num_output=int(model_user_data["num_output"]),
            num_hidden_nodes=json.loads(model_user_data["num_hidden_nodes"]),
            epsilon=0,
        )
        model.network.load_state_dict(network_state_dict)
        model.epsilon = epsilon
        model.num_samples_seen = int(model_user_data["num_samples_seen"])

        return model


class DQNActor:
    def __init__(self, _cfg):
        self._dtype = torch.float
        self.samples_since_update = 0

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config

        rng = np.random.default_rng(config.seed if config.seed is not None else 0)

        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        observation_space = environment_specs.get_observation_space()
        action_space = environment_specs.get_action_space(seed=rng.integers(9999))

        assert isinstance(action_space.gym_space, Discrete)

        model, _, _ = await actor_session.model_registry.retrieve_version(
            DQNModel, config.model_id, config.model_iteration
        )
        model.network.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation = observation_space.deserialize(event.observation.observation)
                if observation.current_player is not None and observation.current_player != actor_session.name:
                    # Not the turn of the agent
                    actor_session.do_action(action_space.serialize(action_space.create()))
                    continue

                if (
                    config.model_update_frequency > 0
                    and self.samples_since_update > 0
                    and self.samples_since_update % config.model_update_frequency == 0
                ):
                    model, _, _ = await actor_session.model_registry.retrieve_version(
                        DQNModel, config.model_id, config.model_iteration
                    )
                    model.network.eval()
                    self.samples_since_update = 0

                if rng.random() < model.epsilon:
                    action = action_space.sample(mask=observation.action_mask)
                else:
                    obs_tensor = torch.tensor(observation.flat_value, dtype=self._dtype)
                    action_probs = model.network(obs_tensor)
                    action_mask = observation.action_mask
                    if action_mask is not None:
                        action_mask_tensor = torch.tensor(action_mask, dtype=self._dtype)
                        large = torch.finfo(self._dtype).max
                        if torch.equal(action_mask_tensor, torch.zeros_like(action_mask_tensor)):
                            log.info("no moves are available, this shouldn't be possible")
                        action_probs = action_probs - large * (1 - action_mask_tensor)
                    discrete_action_tensor = torch.argmax(action_probs)
                    action = action_space.create(value=discrete_action_tensor.item())

                self.samples_since_update += 1
                actor_session.do_action(action_space.serialize(action))


class DQNTraining:
    default_cfg = {
        "seed": 10,
        "num_trials": 5000,
        "num_parallel_trials": 10,
        "learning_rate": 0.00025,
        "buffer_size": 10000,
        "discount_factor": 0.99,
        "target_update_frequency": 500,
        "batch_size": 128,
        "epsilon_schedule_start": 1,
        "epsilon_schedule_end": 0.05,
        "epsilon_schedule_duration_ratio": 0.75,
        "learning_starts": 10000,
        "train_frequency": 10,
        "model_update_frequency": 10,
        "value_network": {"num_hidden_nodes": [128, 64]},
    }

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg

    async def sample_producer_impl(self, sample_producer_session):
        player_actor_params = sample_producer_session.trial_info.parameters.actors[0]

        player_actor_name = player_actor_params.name
        player_environment_specs = EnvironmentSpecs.deserialize(player_actor_params.config.environment_specs)
        player_observation_space = player_environment_specs.get_observation_space()
        player_action_space = player_environment_specs.get_action_space()

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
                player_observation_space.deserialize(actor_sample.observation).flat_value, dtype=self._dtype
            )

            if observation is not None:
                # It's not the first sample, let's check if it is the last
                done = sample.trial_state == cogment.TrialState.ENDED
                sample_producer_session.produce_sample(
                    (
                        observation,
                        next_observation,
                        action,
                        reward,
                        torch.ones(1, dtype=torch.int8) if done else torch.zeros(1, dtype=torch.int8),
                        total_reward,
                    )
                )
                if done:
                    break

            observation = next_observation
            action_value = player_action_space.deserialize(actor_sample.action).value
            action = torch.tensor(action_value, dtype=torch.int64)
            reward = torch.tensor(actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype)
            total_reward += reward.item()

    async def impl(self, run_session):
        # Initializing a model
        model_id = f"{run_session.run_id}_model"

        assert self._environment_specs.num_players == 1
        action_space = self._environment_specs.get_action_space()
        observation_space = self._environment_specs.get_observation_space()
        assert isinstance(action_space.gym_space, Discrete)

        epsilon_schedule = create_linear_schedule(
            self._cfg.epsilon_schedule_start,
            self._cfg.epsilon_schedule_end,
            self._cfg.epsilon_schedule_duration_ratio * self._cfg.num_trials,
        )

        model = DQNModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=utils.flatdim(observation_space.gym_space),
            num_output=utils.flatdim(action_space.gym_space),
            num_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
            epsilon=epsilon_schedule(0),
            dtype=self._dtype,
        )
        serialized_model = DQNModel.serialize_model(model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )

        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
        )

        # Configure the optimizer
        optimizer = torch.optim.Adam(
            model.network.parameters(),
            lr=self._cfg.learning_rate,
        )

        # Initialize the target model
        target_network = copy.deepcopy(model.network)

        replay_buffer = TorchReplayBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=(utils.flatdim(observation_space.gym_space),),
            observation_dtype=self._dtype,
            action_shape=(1,),
            action_dtype=torch.int64,
            reward_dtype=self._dtype,
            seed=self._cfg.seed,
        )

        start_time = time.time()
        total_reward_cum = 0

        for (step_idx, _trial_id, trial_idx, sample,) in run_session.start_and_await_trials(
            trials_id_and_params=[
                (
                    f"{run_session.run_id}_{trial_idx}",
                    cogment.TrialParameters(
                        cog_settings,
                        environment_name="env",
                        environment_implementation=self._environment_specs.implementation,
                        environment_config=EnvironmentConfig(
                            run_id=run_session.run_id,
                            render=False,
                            seed=self._cfg.seed + trial_idx,
                        ),
                        actors=[
                            cogment.ActorParameters(
                                cog_settings,
                                name="player",
                                class_name=PLAYER_ACTOR_CLASS,
                                implementation="actors.dqn.DQNActor",
                                config=AgentConfig(
                                    run_id=run_session.run_id,
                                    seed=self._cfg.seed + trial_idx,
                                    model_id=model_id,
                                    model_version=-1,
                                    model_update_frequency=self._cfg.model_update_frequency,
                                    environment_specs=self._environment_specs.serialize(),
                                ),
                            )
                        ],
                    ),
                )
                for trial_idx in range(self._cfg.num_trials)
            ],
            sample_producer_impl=self.sample_producer_impl,
            num_parallel_trials=self._cfg.num_parallel_trials,
        ):
            (observation, next_observation, action, reward, done, total_reward) = sample
            replay_buffer.add(
                observation=observation, next_observation=next_observation, action=action, reward=reward, done=done
            )

            trial_done = done.item() == 1

            if trial_done:
                run_session.log_metrics(trial_idx=trial_idx, total_reward=total_reward)
                total_reward_cum += total_reward
                if (trial_idx + 1) % 100 == 0:
                    total_reward_avg = total_reward_cum / 100
                    run_session.log_metrics(total_reward_avg=total_reward_avg)
                    total_reward_cum = 0
                    log.info(
                        f"[DQN/{run_session.run_id}] trial #{trial_idx + 1}/{self._cfg.num_trials} done (average total reward = {total_reward_avg})."
                    )

            if (
                step_idx > self._cfg.learning_starts
                and replay_buffer.size() > self._cfg.batch_size
                and step_idx % self._cfg.train_frequency == 0
            ):
                data = replay_buffer.sample(self._cfg.batch_size)

                with torch.no_grad():
                    target_values, _ = target_network(data.next_observation).max(dim=1)
                    td_target = data.reward.flatten() + self._cfg.discount_factor * target_values * (
                        1 - data.done.flatten()
                    )

                action_values = model.network(data.observation).gather(1, data.action).squeeze()
                loss = torch.nn.functional.mse_loss(td_target, action_values)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the epsilon
                model.epsilon = epsilon_schedule(trial_idx)

                # Update the version info
                model.num_samples_seen += data.size()

                if step_idx % self._cfg.target_update_frequency == 0:
                    target_network.load_state_dict(model.network.state_dict())

                serialized_model = DQNModel.serialize_model(model)
                iteration_info = await run_session.model_registry.publish_model(
                    name=model_id,
                    model=serialized_model,
                )

                if step_idx % 100 == 0:
                    end_time = time.time()
                    steps_per_seconds = 100 / (end_time - start_time)
                    start_time = end_time
                    run_session.log_metrics(
                        model_version_number=version_info["version_number"],
                        loss=loss.item(),
                        q_values=action_values.mean().item(),
                        batch_avg_reward=data.reward.mean().item(),
                        epsilon=model.epsilon,
                        steps_per_seconds=steps_per_seconds,
                    )

        serialized_model = DQNModel.serialize_model(model)
        iteration_info = await run_session.model_registry.store_model(
            name=model_id,
            model=serialized_model,
        )
