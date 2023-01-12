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

import copy
import logging
import math
import time
from ast import literal_eval

import cogment
import numpy as np
import torch

from cogment_verse import Model, TorchReplayBuffer  # pylint: disable=abstract-class-instantiated
from cogment_verse.specs import (
    HUMAN_ACTOR_IMPL,
    PLAYER_ACTOR_CLASS,
    WEB_ACTOR_NAME,
    AgentConfig,
    EnvironmentConfig,
    PlayerAction,
    SpaceValue,
    cog_settings,
    flatten,
    flattened_dimensions,
)

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)


def create_linear_schedule(start, end, duration):
    slope = (end - start) / duration

    def compute_value(t):
        return max(slope * t + start, end)

    return compute_value


def initalize_layer(layer: torch.nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(torch.nn.Module):
    """Policy and Value networks for Atari games"""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.shared_network = torch.nn.Sequential(
            initalize_layer(torch.nn.Conv2d(6, 32, 8, stride=4)),
            torch.nn.ReLU(),
            initalize_layer(torch.nn.Conv2d(32, 64, 3, stride=2)),
            torch.nn.ReLU(),
            initalize_layer(torch.nn.Conv2d(64, 64, 3, stride=1)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            initalize_layer(torch.nn.Linear(64 * 7 * 7, 512)),
            torch.nn.ReLU(),
            initalize_layer(torch.nn.Linear(512, num_actions), 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation / 255.0
        return self.shared_network(observation)


class DQNModel(Model):
    def __init__(
        self,
        model_id: int,
        environment_implementation: str,
        num_actions: int,
        input_shape: tuple,
        epsilon: float,
        learning_rate: float = 3e-4,
        dtype=torch.float32,
        device: str = "cpu",
        version_number=0,
    ):
        super().__init__(model_id=model_id, version_number=version_number)
        self._dtype = dtype
        self._environment_implementation = environment_implementation
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.device = device
        self.network = QNetwork(self.num_actions)
        self.network.to(torch.device(self.device))

        # Optimizer
        self.network_optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)

        # version user data
        self.num_samples_seen = 0

    def get_model_user_data(self):
        return {
            "environment_implementation": self._environment_implementation,
            "num_actions": self.num_actions,
            "input_shape": self.input_shape,
            "device": self.device,
        }

    def save(self, model_data_f):
        torch.save((self.network.state_dict(), self.epsilon), model_data_f)

        return {"num_samples_seen": self.num_samples_seen}

    @classmethod
    def load(cls, model_id, version_number, model_user_data, version_user_data, model_data_f):
        # Create the model instance
        model = DQNModel(
            model_id=model_id,
            version_number=version_number,
            environment_implementation=model_user_data["environment_implementation"],
            num_actions=int(model_user_data["num_actions"]),
            input_shape=literal_eval(model_user_data["input_shape"]),
            epsilon=0,
            device=model_user_data["device"],
        )

        # Load the saved states
        (network_state_dict, epsilon) = torch.load(model_data_f)
        model.network.load_state_dict(network_state_dict)
        model.epsilon = epsilon

        # Load version data
        model.num_samples_seen = int(version_user_data["num_samples_seen"])

        return model


class DQNActor:
    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        # Start a session
        actor_session.start()
        config = actor_session.config
        assert len(config.environment_specs.action_space.properties) == 1

        # Get observation and action space
        observation_space = config.environment_specs.observation_space

        rng = np.random.default_rng(config.seed if config.seed is not None else 0)

        model, _, _ = await actor_session.model_registry.retrieve_version(
            DQNModel, config.model_id, config.model_version
        )
        obs_shape = model.input_shape[::-1]

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                if (
                    event.observation.observation.HasField("current_player")
                    and event.observation.observation.current_player != actor_session.name
                ):
                    # Not the turn of the agent
                    if event.observation.observation.current_player == WEB_ACTOR_NAME:
                        actor_session.do_action(PlayerAction())
                    else:
                        actor_session.do_action(PlayerAction())
                    continue

                if (
                    config.model_version == -1
                    and config.model_update_frequency > 0
                    and actor_session.get_tick_id() % config.model_update_frequency == 0
                ):
                    model, _, _ = await actor_session.model_registry.retrieve_version(
                        DQNModel, config.model_id, config.model_version
                    )
                if rng.random() < model.epsilon:
                    action = np.random.choice(model.num_actions, 1, replace=False)
                    action_value = SpaceValue(properties=[SpaceValue.PropertyValue(discrete=action[0])])
                else:

                    obs_flat = torch.tensor(
                        flatten(observation_space, event.observation.observation.value), dtype=self._dtype
                    ).reshape(obs_shape)
                    obs_tensor = torch.unsqueeze(obs_flat.permute((2, 0, 1)), dim=0).to(torch.device(model.device))
                    with torch.no_grad():
                        action_probs = model.network(obs_tensor)
                        action_tensor = torch.argmax(action_probs)
                    action_value = SpaceValue(properties=[SpaceValue.PropertyValue(discrete=action_tensor.item())])

                actor_session.do_action(PlayerAction(value=action_value))


class DQNSelfPlayTraining:
    default_cfg = {
        "seed": 10,
        "num_epochs": 50,
        "epoch_num_training_trials": 100,
        "hill_training_trials_ratio": 0,
        "epoch_num_validation_trials": 10,
        "num_parallel_trials": 10,
        "learning_rate": 0.00025,
        "buffer_size": 100_000,
        "discount_factor": 0.99,
        "target_update_frequency": 500,
        "batch_size": 128,
        "epsilon_schedule_start": 1,
        "epsilon_schedule_end": 0.05,
        "epsilon_schedule_duration_ratio": 0.75,
        "learning_starts": 10000,
        "train_frequency": 10,
        "model_update_frequency": 10,
        "device": "cpu",
        "image_size": [6, 84, 84],
    }

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32
        self._environment_specs = environment_specs
        self._cfg = cfg
        self._rng = np.random.default_rng(self._cfg.seed)
        self.model = DQNModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            num_actions=flattened_dimensions(self._environment_specs.action_space),
            input_shape=tuple(self._cfg.image_size),
            epsilon=0.1,
            learning_rate=self._cfg.learning_rate,
            device=self._cfg.device,
        )

    async def sample_producer_impl(self, sample_producer_session):
        players_params = {
            actor_params.name: actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == PLAYER_ACTOR_CLASS
        }

        players_partial_sample = {
            actor_params.name: {"observation": None, "action": None, "reward": None, "total_reward": 0}
            for actor_params in players_params.values()
        }
        obs_shape = tuple(self._cfg.image_size)[::-1]

        # Let's start with any player actor
        current_player_actor = next(iter(players_params.keys()))
        total_steps = 0
        async for sample in sample_producer_session.all_trial_samples():
            previous_player_actor_sample = sample.actors_data[current_player_actor]
            if previous_player_actor_sample.observation is None:
                # This can happen when there is several "end-of-trial" samples
                continue

            current_player_actor = previous_player_actor_sample.observation.current_player
            current_player_params = players_params[current_player_actor]
            current_player_partial_sample = players_partial_sample[current_player_actor]

            current_player_sample = sample.actors_data[current_player_actor]

            next_observation_flat = torch.tensor(
                flatten(
                    current_player_params.config.environment_specs.observation_space,
                    current_player_sample.observation.value,
                ),
                dtype=self._dtype,
            ).reshape(obs_shape)
            next_observation = next_observation_flat.permute((2, 0, 1))

            if current_player_partial_sample["observation"] is not None:
                # It's not the first sample, let's check if it is the last
                done = sample.trial_state == cogment.TrialState.ENDED
                sample_producer_session.produce_sample(
                    (
                        current_player_actor,
                        current_player_partial_sample["observation"],
                        next_observation,
                        current_player_partial_sample["action"],
                        current_player_partial_sample["reward"],
                        torch.ones(1, dtype=torch.int8) if done else torch.zeros(1, dtype=torch.int8),
                        {
                            actor_name: partial_sample["total_reward"]
                            for actor_name, partial_sample in players_partial_sample.items()
                        },
                        total_steps,
                    )
                )
                if done:
                    total_steps = 0
                    break

            current_player_partial_sample["observation"] = next_observation
            action_value = current_player_sample.action.value
            current_player_partial_sample["action"] = torch.tensor(
                action_value.properties[0].discrete if len(action_value.properties) > 0 else 0, dtype=torch.int64
            )
            total_steps += 1
            for player_actor in players_params.keys():
                player_partial_sample = players_partial_sample[player_actor]
                player_partial_sample["reward"] = torch.tensor(
                    sample.actors_data[player_actor].reward
                    if sample.actors_data[player_actor].reward is not None
                    else 0,
                    dtype=self._dtype,
                )
                player_partial_sample["total_reward"] += player_partial_sample["reward"].item()

    async def impl(self, run_session):
        # Initializing a model
        model_id = f"{run_session.run_id}_model"

        # assert self._environment_specs.num_players == 2
        # assert len(self._environment_specs.action_space.properties) == 1
        # assert self._environment_specs.action_space.properties[0].WhichOneof("type") == "discrete"

        self.model.model_id = model_id
        _, version_info = await run_session.model_registry.publish_initial_version(self.model)
        run_session.log_params(
            self._cfg,
            model_id=model_id,
            environment_implementation=self._environment_specs.implementation,
        )

        # Greedy epsilon
        epsilon_schedule = create_linear_schedule(
            self._cfg.epsilon_schedule_start,
            self._cfg.epsilon_schedule_end,
            self._cfg.epsilon_schedule_duration_ratio * self._cfg.num_epochs * self._cfg.epoch_num_training_trials,
        )

        # Initialize the target model
        target_network = copy.deepcopy(self.model.network)

        replay_buffer = TorchReplayBuffer(
            capacity=self._cfg.buffer_size,
            observation_shape=tuple(self._cfg.image_size),
            observation_dtype=self._dtype,
            action_shape=(1,),
            action_dtype=torch.int64,
            reward_dtype=self._dtype,
            seed=self._rng.integers(9999),
        )

        def create_actor_params(name, version_number=-1, human=False):
            if human:
                return cogment.ActorParameters(
                    cog_settings,
                    name=WEB_ACTOR_NAME,
                    class_name=PLAYER_ACTOR_CLASS,
                    implementation=HUMAN_ACTOR_IMPL,
                    config=AgentConfig(
                        run_id=run_session.run_id,
                        environment_specs=self._environment_specs,
                    ),
                )
            return cogment.ActorParameters(
                cog_settings,
                name=name,
                class_name=PLAYER_ACTOR_CLASS,
                implementation="actors.dqn_atari_pz.DQNActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    seed=self._rng.integers(9999),
                    model_id=model_id,
                    model_version=version_number,
                    model_update_frequency=self._cfg.model_update_frequency,
                    environment_specs=self._environment_specs,
                ),
            )

        def create_trials_params(p1_params, p2_params):
            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id,
                    render=HUMAN_ACTOR_IMPL in (p1_params.implementation, p2_params.implementation),
                    seed=self._rng.integers(9999),
                ),
                actors=[p1_params, p2_params],
            )

        hill_training_trial_period = (
            math.floor(1 / self._cfg.hill_training_trials_ratio) if self._cfg.hill_training_trials_ratio > 0 else 0
        )

        running_avg_len = 0
        for epoch_idx in range(self._cfg.num_epochs):
            start_time = time.time()

            # Self training trials
            for (step_idx, _trial_id, trial_idx, sample,) in run_session.start_and_await_trials(
                trials_id_and_params=[
                    (
                        f"{run_session.run_id}_{epoch_idx}_t_{trial_idx}",
                        create_trials_params(
                            p1_params=create_actor_params(
                                "first_0",
                                human=(
                                    hill_training_trial_period > 0
                                    and trial_idx % (hill_training_trial_period * 2) == 0
                                    and trial_idx % hill_training_trial_period == 0
                                ),
                            ),
                            p2_params=create_actor_params(
                                "second_0",
                                human=(
                                    hill_training_trial_period > 0
                                    and trial_idx % (hill_training_trial_period * 2) != 0
                                    and trial_idx % hill_training_trial_period == 0
                                ),
                            ),
                        ),
                    )
                    for trial_idx in range(self._cfg.epoch_num_training_trials)
                ],
                sample_producer_impl=self.sample_producer_impl,
                num_parallel_trials=self._cfg.num_parallel_trials,
            ):
                (_actor_name, observation, next_observation, action, reward, done, total_rewards, total_steps) = sample
                replay_buffer.add(
                    observation=observation, next_observation=next_observation, action=action, reward=reward, done=done
                )

                trial_done = done.item() == 1

                if trial_done:
                    run_session.log_metrics(training_total_reward=sum(total_rewards.values()), episode_len=total_steps)
                    running_avg_len += total_steps
                    if (step_idx + 1) % 100 == 0:
                        run_session.log_metrics(episode_len=running_avg_len / 100)
                        running_avg_len = 0

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

                    action_values = self.model.network(data.observation).gather(1, data.action).squeeze()
                    loss = torch.nn.functional.mse_loss(td_target, action_values)

                    # optimize the model
                    self.model.network_optimizer.zero_grad()
                    loss.backward()
                    self.model.network_optimizer.step()

                    # Update the epsilon
                    self.model.epsilon = epsilon_schedule(epoch_idx * self._cfg.epoch_num_training_trials + trial_idx)

                    # Update the version info
                    self.model.num_samples_seen += data.size()

                    if step_idx % self._cfg.target_update_frequency == 0:
                        target_network.load_state_dict(self.model.network.state_dict())
                    log.info(f"Epoch #{epoch_idx} | Loss {loss.item():.4f}")

                    version_info = await run_session.model_registry.publish_version(self.model)

                    if step_idx % 100 == 0:
                        end_time = time.time()
                        steps_per_seconds = 100 / (end_time - start_time)
                        start_time = end_time
                        run_session.log_metrics(
                            model_version_number=version_info["version_number"],
                            loss=loss.item(),
                            q_values=action_values.mean().item(),
                            epsilon=self.model.epsilon,
                            steps_per_seconds=steps_per_seconds,
                        )

            version_info = await run_session.model_registry.publish_version(self.model, archived=True)
