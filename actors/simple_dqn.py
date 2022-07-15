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
import copy
import time
import json
import math

import cogment
import torch
import numpy as np

from cogment_verse.specs import (
    AgentConfig,
    cog_settings,
    EnvironmentConfig,
    flatten,
    flattened_dimensions,
    flatten_mask,
    PLAYER_ACTOR_CLASS,
    PlayerAction,
    SpaceValue,
    sample_space,
    WEB_ACTOR_NAME,
    HUMAN_ACTOR_IMPL,
)

from cogment_verse import Model, TorchReplayBuffer


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
        version_number=0,
    ):
        super().__init__(model_id, version_number)
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
            "environment_implementation": self._environment_implementation,
            "num_input": self._num_input,
            "num_output": self._num_output,
            "num_hidden_nodes": json.dumps(self._num_hidden_nodes),
        }

    def save(self, model_data_f):
        torch.save((self.network.state_dict(), self.epsilon), model_data_f)

        return {"num_samples_seen": self.num_samples_seen}

    @classmethod
    def load(cls, model_id, version_number, model_user_data, version_user_data, model_data_f):
        # Create the model instance
        model = SimpleDQNModel(
            model_id=model_id,
            version_number=version_number,
            environment_implementation=model_user_data["environment_implementation"],
            num_input=int(model_user_data["num_input"]),
            num_output=int(model_user_data["num_output"]),
            num_hidden_nodes=json.loads(model_user_data["num_hidden_nodes"]),
            epsilon=0,
        )

        # Load the saved states
        (network_state_dict, epsilon) = torch.load(model_data_f)
        model.network.load_state_dict(network_state_dict)
        model.epsilon = epsilon

        # Load version data
        model.num_samples_seen = int(version_user_data["num_samples_seen"])

        return model


class SimpleDQNActor:
    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config

        assert len(config.environment_specs.action_space.properties) == 1
        assert config.environment_specs.action_space.properties[0].WhichOneof("type") == "discrete"

        observation_space = config.environment_specs.observation_space
        action_space = config.environment_specs.action_space

        rng = np.random.default_rng(config.seed if config.seed is not None else 0)

        model, _, _ = await actor_session.model_registry.retrieve_version(
            SimpleDQNModel, config.model_id, config.model_version
        )
        model.network.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                if (
                    event.observation.observation.HasField("current_player")
                    and event.observation.observation.current_player != actor_session.name
                ):
                    # Not the turn of the agent
                    actor_session.do_action(PlayerAction())
                    continue

                if (
                    config.model_version == -1
                    and config.model_update_frequency > 0
                    and actor_session.get_tick_id() % config.model_update_frequency == 0
                ):
                    model, _, _ = await actor_session.model_registry.retrieve_version(
                        SimpleDQNModel, config.model_id, config.model_version
                    )
                    model.network.eval()
                if rng.random() < model.epsilon:
                    [action_value] = sample_space(action_space, rng=rng, mask=event.observation.observation.action_mask)
                else:
                    obs_tensor = torch.tensor(
                        flatten(observation_space, event.observation.observation.value), dtype=self._dtype
                    )
                    action_probs = model.network(obs_tensor)
                    if event.observation.observation.HasField("action_mask"):
                        action_mask = torch.tensor(
                            flatten_mask(action_space, event.observation.observation.action_mask), dtype=self._dtype
                        )
                        large = torch.finfo(self._dtype).max
                        if torch.equal(action_mask, torch.zeros_like(action_mask)):
                            log.info("no moves are available, this shouldn't be possible")
                        action_probs = action_probs - large * (1 - action_mask)
                    discrete_action_tensor = torch.argmax(action_probs)
                    action_value = SpaceValue(
                        properties=[SpaceValue.PropertyValue(discrete=discrete_action_tensor.item())]
                    )

                actor_session.do_action(PlayerAction(value=action_value))


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
        player_observation_space = player_actor_params.config.environment_specs.observation_space

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
            action_value = actor_sample.action.value
            action = torch.tensor(
                action_value.properties[0].discrete if len(action_value.properties) > 0 else 0, dtype=torch.int64
            )
            reward = torch.tensor(actor_sample.reward if actor_sample.reward is not None else 0, dtype=self._dtype)
            total_reward += reward.item()

    async def impl(self, run_session):
        # Initializing a model
        model_id = f"{run_session.run_id}_model"

        assert self._environment_specs.num_players == 1
        assert len(self._environment_specs.action_space.properties) == 1
        assert self._environment_specs.action_space.properties[0].WhichOneof("type") == "discrete"

        epsilon_schedule = create_linear_schedule(
            self._cfg.epsilon_schedule_start,
            self._cfg.epsilon_schedule_end,
            self._cfg.epsilon_schedule_duration_ratio * self._cfg.num_trials,
        )

        model = SimpleDQNModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=flattened_dimensions(self._environment_specs.observation_space),
            num_output=flattened_dimensions(self._environment_specs.action_space),
            num_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
            epsilon=epsilon_schedule(0),
            dtype=self._dtype,
        )
        _model_info, version_info = await run_session.model_registry.publish_initial_version(model)

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
            observation_shape=(flattened_dimensions(self._environment_specs.observation_space),),
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
                                implementation="actors.simple_dqn.SimpleDQNActor",
                                config=AgentConfig(
                                    run_id=run_session.run_id,
                                    seed=self._cfg.seed + trial_idx,
                                    model_id=model_id,
                                    model_version=-1,
                                    model_update_frequency=self._cfg.model_update_frequency,
                                    environment_specs=self._environment_specs,
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
                run_session.log_metrics(trial_idx, total_reward=total_reward)
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

                version_info = await run_session.model_registry.publish_version(model)

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

        version_info = await run_session.model_registry.publish_version(model, archived=True)


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
            actor_params.name: actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == PLAYER_ACTOR_CLASS
        }

        players_partial_sample = {
            actor_params.name: {"observation": None, "action": None, "reward": None, "total_reward": 0}
            for actor_params in players_params.values()
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
                flatten(
                    current_player_params.config.environment_specs.observation_space,
                    current_player_sample.observation.value,
                ),
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
            action_value = current_player_sample.action.value
            current_player_partial_sample["action"] = torch.tensor(
                action_value.properties[0].discrete if len(action_value.properties) > 0 else 0, dtype=torch.int64
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
        assert len(self._environment_specs.action_space.properties) == 1
        assert self._environment_specs.action_space.properties[0].WhichOneof("type") == "discrete"

        epsilon_schedule = create_linear_schedule(
            self._cfg.epsilon_schedule_start,
            self._cfg.epsilon_schedule_end,
            self._cfg.epsilon_schedule_duration_ratio * self._cfg.num_epochs * self._cfg.epoch_num_training_trials,
        )

        model = SimpleDQNModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=flattened_dimensions(self._environment_specs.observation_space),
            num_output=flattened_dimensions(self._environment_specs.action_space),
            num_hidden_nodes=self._cfg.value_network.num_hidden_nodes,
            epsilon=epsilon_schedule(0),
            dtype=self._dtype,
        )
        _model_info, version_info = await run_session.model_registry.publish_initial_version(model)

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
            observation_shape=(flattened_dimensions(self._environment_specs.observation_space),),
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
                implementation="actors.simple_dqn.SimpleDQNActor"
                if version_number is not None
                else "actors.random_actor.RandomActor",
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

        previous_epoch_version_number = None
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

                    version_info = await run_session.model_registry.publish_version(model)

                    if step_idx % 100 == 0:
                        end_time = time.time()
                        steps_per_seconds = 100 / (end_time - start_time)
                        start_time = end_time
                        run_session.log_metrics(
                            model_version_number=version_info["version_number"],
                            loss=loss.item(),
                            q_values=action_values.mean().item(),
                            epsilon=model.epsilon,
                            steps_per_seconds=steps_per_seconds,
                        )

            version_info = await run_session.model_registry.publish_version(model, archived=True)

            # Validation trials
            cum_total_reward = 0
            num_ties = 0
            for (step_idx, _trial_id, trial_idx, sample,) in run_session.start_and_await_trials(
                trials_id_and_params=[
                    (
                        f"{run_session.run_id}_{epoch_idx}_v_{trial_idx}",
                        create_trials_params(
                            p1_params=create_actor_params("reference", previous_epoch_version_number)
                            if trial_idx % 2 == 0
                            else create_actor_params("validated", version_info["version_number"]),
                            p2_params=create_actor_params("reference", previous_epoch_version_number)
                            if trial_idx % 2 == 1
                            else create_actor_params("validated", version_info["version_number"]),
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
            validation_version_number = version_info["version_number"]
            run_session.log_metrics(
                validation_avg_total_reward=avg_total_reward,
                validation_ties_ratio=ties_ratio,
                validation_version_number=validation_version_number,
            )
            if previous_epoch_version_number is not None:
                run_session.log_metrics(
                    reference_version_number=previous_epoch_version_number,
                )
            log.info(
                f"[SimpleDQN/{run_session.run_id}] epoch #{epoch_idx + 1}/{self._cfg.num_epochs} done - "
                + f"[{model.model_id}@v{validation_version_number}] avg total reward = {avg_total_reward}, ties ratio = {ties_ratio}"
            )
            previous_epoch_version_number = validation_version_number
