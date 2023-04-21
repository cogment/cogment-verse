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

from __future__ import annotations

import copy
import io
import json
import logging
import math
import time

import cogment
import numpy as np
import torch
from gym.spaces import Discrete, utils

from cogment_verse import Model, TorchReplayBuffer  # pylint: disable=abstract-class-instantiated
from cogment_verse.constants import HUMAN_ACTOR_IMPL, WEB_ACTOR_NAME, ActorClass
from cogment_verse.specs import AgentConfig, EnvironmentConfig, EnvironmentSpecs, cog_settings

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)


def create_linear_schedule(start, end, duration):
    slope = (end - start) / duration

    def compute_value(t):
        return max(slope * t + start, end)

    return compute_value


class SimpleDQNModel(Model):
    def __init__(
        self,
        model_id,
        environment_implementation,
        num_input,
        num_output,
        num_hidden_nodes,
        epsilon,
        dtype=torch.float,
        iteration=0,
    ):
        super().__init__(model_id, iteration)
        self._dtype = dtype
        self._environment_implementation = environment_implementation
        self._num_input = num_input
        self._num_output = num_output
        self._num_hidden_nodes = list(num_hidden_nodes)

        self.epsilon = epsilon
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self._num_input, self._num_hidden_nodes[0]),
            torch.nn.ReLU(),
            *[
                layer
                for hidden_node_idx in range(len(self._num_hidden_nodes) - 1)
                for layer in [
                    torch.nn.Linear(self._num_hidden_nodes[hidden_node_idx], self._num_hidden_nodes[-1]),
                    torch.nn.ReLU(),
                ]
            ],
            torch.nn.Linear(self._num_hidden_nodes[-1], self._num_output),
        )

        # version user data
        self.num_samples_seen = 0

    def get_model_user_data(self):
        return {
            "model_id": self.model_id,
            "iteration": self.iteration,
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
    def deserialize_model(cls, serialized_model) -> SimpleDQNModel:
        stream = io.BytesIO(serialized_model)
        (network_state_dict, epsilon, model_user_data) = torch.load(stream)

        model = cls(
            model_id=model_user_data["model_id"],
            iteration=model_user_data["iteration"],
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


class SimpleDQNActor:
    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        return [ActorClass.PLAYER.value]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config

        rng = np.random.default_rng(config.seed if config.seed is not None else 0)

        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        observation_space = environment_specs.get_observation_space()
        action_space = environment_specs.get_action_space(seed=rng.integers(9999))

        assert isinstance(action_space.gym_space, Discrete)

        # Get model
        if config.model_iteration == -1:
            latest_model = await actor_session.model_registry.track_latest_model(
                name=config.model_id, deserialize_func=SimpleDQNModel.deserialize_model
            )
            model, _ = await latest_model.get()
        else:
            serialized_model = await actor_session.model_registry.retrieve_model(
                config.model_id, config.model_iteration
            )
            model = SimpleDQNModel.deserialize_model(serialized_model)

        model.network.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation = observation_space.deserialize(event.observation.observation)
                if observation.current_player is not None and observation.current_player != actor_session.name:
                    # Not the turn of the agent
                    actor_session.do_action(action_space.serialize(action_space.create()))
                    continue

                if (
                    config.model_iteration == -1
                    and config.model_update_frequency > 0
                    and actor_session.get_tick_id() % config.model_update_frequency == 0
                ):
                    # Get model
                    if config.model_iteration == -1:
                        latest_model = await actor_session.model_registry.track_latest_model(
                            name=config.model_id, deserialize_func=SimpleDQNModel.deserialize_model
                        )
                        model, _ = await latest_model.get()
                    else:
                        serialized_model = await actor_session.model_registry.retrieve_model(
                            config.model_id, config.model_iteration
                        )
                        model = SimpleDQNModel.deserialize_model(serialized_model)

                    model.network.eval()

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

                actor_session.do_action(action_space.serialize(action))


class SimpleDQNTraining:
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

        model = SimpleDQNModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=utils.flatdim(observation_space.gym_space),
            num_output=utils.flatdim(action_space.gym_space),
            num_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
            epsilon=epsilon_schedule(0),
            dtype=self._dtype,
        )

        serialized_model = SimpleDQNModel.serialize_model(model)
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
                                class_name=ActorClass.PLAYER.value,
                                implementation="actors.simple_dqn.SimpleDQNActor",
                                config=AgentConfig(
                                    run_id=run_session.run_id,
                                    seed=self._cfg.seed + trial_idx,
                                    model_id=model_id,
                                    model_iteration=-1,
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
                        f"[SimpleDQN/{run_session.run_id}] trial #{trial_idx + 1}/{self._cfg.num_trials} done (average total reward = {total_reward_avg})."
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

                serialized_model = SimpleDQNModel.serialize_model(model)
                iteration_info = await run_session.model_registry.publish_model(
                    name=model_id,
                    model=serialized_model,
                )

                if step_idx % 100 == 0:
                    end_time = time.time()
                    steps_per_seconds = 100 / (end_time - start_time)
                    start_time = end_time
                    run_session.log_metrics(
                        model_iteration=iteration_info.iteration,
                        loss=loss.item(),
                        q_values=action_values.mean().item(),
                        batch_avg_reward=data.reward.mean().item(),
                        epsilon=model.epsilon,
                        steps_per_seconds=steps_per_seconds,
                    )

        serialized_model = SimpleDQNModel.serialize_model(model)
        iteration_info = await run_session.model_registry.store_model(
            name=model_id,
            model=serialized_model,
        )


class SimpleDQNSelfPlayTraining:
    default_cfg = {
        "seed": 10,
        "num_epochs": 50,
        "epoch_num_training_trials": 100,
        "hill_training_trials_ratio": 0,
        "epoch_num_validation_trials": 10,
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
        self._rng = np.random.default_rng(self._cfg.seed)

    async def sample_producer_impl(self, sample_producer_session):
        players_params = {
            actor_params.name: {
                "params": actor_params,
                "observation_space": EnvironmentSpecs.deserialize(
                    actor_params.config.environment_specs
                ).get_observation_space(),
                "action_space": EnvironmentSpecs.deserialize(actor_params.config.environment_specs).get_action_space(),
            }
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == ActorClass.PLAYER.value
        }

        players_partial_sample = {
            player_params["params"].name: {"observation": None, "action": None, "reward": None, "total_reward": 0}
            for player_params in players_params.values()
        }

        # Let's start with any player actor
        current_player_actor = next(iter(players_params.keys()))
        async for sample in sample_producer_session.all_trial_samples():
            previous_player_actor_sample = sample.actors_data[current_player_actor]
            if previous_player_actor_sample.observation is None:
                # This can happen when there is several "end-of-trial" samples
                continue

            current_player_actor = previous_player_actor_sample.observation.current_player
            current_player_params = players_params[current_player_actor]
            current_player_partial_sample = players_partial_sample[current_player_actor]

            current_player_sample = sample.actors_data[current_player_actor]

            next_observation = torch.tensor(
                current_player_params["observation_space"].deserialize(current_player_sample.observation).flat_value,
                dtype=self._dtype,
            )

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
                    )
                )
                if done:
                    break

            current_player_partial_sample["observation"] = next_observation
            action_value = current_player_params["action_space"].deserialize(current_player_sample.action).value
            current_player_partial_sample["action"] = torch.tensor(
                action_value,
                dtype=torch.int64,
            )
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

        assert self._environment_specs.num_players == 2
        action_space = self._environment_specs.get_action_space()
        observation_space = self._environment_specs.get_observation_space()
        assert isinstance(action_space.gym_space, Discrete)

        epsilon_schedule = create_linear_schedule(
            self._cfg.epsilon_schedule_start,
            self._cfg.epsilon_schedule_end,
            self._cfg.epsilon_schedule_duration_ratio * self._cfg.num_epochs * self._cfg.epoch_num_training_trials,
        )

        model = SimpleDQNModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=utils.flatdim(observation_space.gym_space),
            num_output=utils.flatdim(action_space.gym_space),
            num_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
            epsilon=epsilon_schedule(0),
            dtype=self._dtype,
        )

        serialized_model = SimpleDQNModel.serialize_model(model)
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
            seed=self._rng.integers(9999),
        )

        def create_actor_params(name, iteration=-1, human=False):
            if human:
                return cogment.ActorParameters(
                    cog_settings,
                    name=WEB_ACTOR_NAME,
                    class_name=ActorClass.PLAYER.value,
                    implementation=HUMAN_ACTOR_IMPL,
                    config=AgentConfig(
                        run_id=run_session.run_id,
                        environment_specs=self._environment_specs.serialize(),
                    ),
                )
            return cogment.ActorParameters(
                cog_settings,
                name=name,
                class_name=ActorClass.PLAYER.value,
                implementation="actors.simple_dqn.SimpleDQNActor"
                if iteration is not None
                else "actors.random_actor.RandomActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    seed=self._rng.integers(9999),
                    model_id=model_id,
                    model_iteration=iteration,
                    model_update_frequency=self._cfg.model_update_frequency,
                    environment_specs=self._environment_specs.serialize(),
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

        previous_epoch_iteration = None
        for epoch_idx in range(self._cfg.num_epochs):
            start_time = time.time()

            # Self training trials
            for (step_idx, _trial_id, trial_idx, sample,) in run_session.start_and_await_trials(
                trials_id_and_params=[
                    (
                        f"{run_session.run_id}_{epoch_idx}_t_{trial_idx}",
                        create_trials_params(
                            p1_params=create_actor_params(
                                "player_1",
                                human=(
                                    hill_training_trial_period > 0
                                    and trial_idx % (hill_training_trial_period * 2) == 0
                                    and trial_idx % hill_training_trial_period == 0
                                ),
                            ),
                            p2_params=create_actor_params(
                                "player_2",
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
                (_actor_name, observation, next_observation, action, reward, done, total_rewards) = sample
                replay_buffer.add(
                    observation=observation, next_observation=next_observation, action=action, reward=reward, done=done
                )

                trial_done = done.item() == 1

                if trial_done:
                    run_session.log_metrics(training_total_reward=sum(total_rewards.values()))

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
                    model.epsilon = epsilon_schedule(epoch_idx * self._cfg.epoch_num_training_trials + trial_idx)

                    # Update the version info
                    model.num_samples_seen += data.size()

                    if step_idx % self._cfg.target_update_frequency == 0:
                        target_network.load_state_dict(model.network.state_dict())

                    serialized_model = SimpleDQNModel.serialize_model(model)
                    iteration_info = await run_session.model_registry.publish_model(
                        name=model_id,
                        model=serialized_model,
                    )

                    if step_idx % 100 == 0:
                        end_time = time.time()
                        steps_per_seconds = 100 / (end_time - start_time)
                        start_time = end_time
                        run_session.log_metrics(
                            model_iteration=iteration_info.iteration,
                            loss=loss.item(),
                            q_values=action_values.mean().item(),
                            epsilon=model.epsilon,
                            steps_per_seconds=steps_per_seconds,
                        )

            serialized_model = SimpleDQNModel.serialize_model(model)
            iteration_info = await run_session.model_registry.store_model(
                name=model_id,
                model=serialized_model,
            )

            # Validation trials
            cum_total_reward = 0
            num_ties = 0
            for (step_idx, _trial_id, trial_idx, sample,) in run_session.start_and_await_trials(
                trials_id_and_params=[
                    (
                        f"{run_session.run_id}_{epoch_idx}_v_{trial_idx}",
                        create_trials_params(
                            p1_params=create_actor_params("reference", previous_epoch_iteration)
                            if trial_idx % 2 == 0
                            else create_actor_params("validated", iteration_info.iteration),
                            p2_params=create_actor_params("reference", previous_epoch_iteration)
                            if trial_idx % 2 == 1
                            else create_actor_params("validated", iteration_info.iteration),
                        ),
                    )
                    for trial_idx in range(self._cfg.epoch_num_validation_trials)
                ],
                sample_producer_impl=self.sample_producer_impl,
                num_parallel_trials=self._cfg.num_parallel_trials,
            ):
                (_actor_name, _observation, _next_observation, _action, _reward, done, total_rewards) = sample

                trial_done = done.item() == 1

                if trial_done:
                    cum_total_reward += total_rewards["validated"]
                    if total_rewards["validated"] == 0:
                        num_ties += 1

            avg_total_reward = cum_total_reward / self._cfg.epoch_num_validation_trials
            ties_ratio = num_ties / self._cfg.epoch_num_validation_trials
            validation_iteration = iteration_info.iteration
            run_session.log_metrics(
                validation_avg_total_reward=avg_total_reward,
                validation_ties_ratio=ties_ratio,
                validation_iteration=validation_iteration,
            )
            if previous_epoch_iteration is not None:
                run_session.log_metrics(
                    reference_iteration=previous_epoch_iteration,
                )
            log.info(
                f"[SimpleDQN/{run_session.run_id}] epoch #{epoch_idx + 1}/{self._cfg.num_epochs} done - "
                + f"[{model.model_id}@v{validation_iteration}] avg total reward = {avg_total_reward}, ties ratio = {ties_ratio}"
            )
            previous_epoch_iteration = validation_iteration
