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
import numpy as np

import cogment
import torch

from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.qnet_heads import DQNNetwork
from hive.agents.qnets.utils import (
    InitializationFn,
    calculate_output_dim,
    create_init_weights_fn,
)

from hive.utils.schedule import (
    LinearSchedule,
    PeriodicSchedule,
    Schedule,
    SwitchSchedule,
)
from hive.utils.utils import LossFn, OptimizerFn, create_folder, seeder

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

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)


def create_linear_schedule(start, end, duration):
    slope = (end - start) / duration

    def compute_value(t):
        return max(slope * t + start, end)

    return compute_value


class HiveDQNModel(Model):
    def __init__(
        self,
        model_id,
        environment_implementation,
        num_input,
        num_output,
        version_number=0,
        representation_net: FunctionApproximator,
        epsilon_schedule: Schedule = None,
    ):
        super().__init__(model_id, version_number)
        self._environment_implementation = environment_implementation
        self._num_input = num_input
        self._num_output = num_output
        self._init_fn = create_init_weights_fn(init_fn)
        self.network = self.create_q_networks(representation_net)
        self.device = "cpu"

        if epsilon_schedule is None:
            self._epsilon_schedule = LinearSchedule(1, 0.1, 100000)
        else:
            self._epsilon_schedule = epsilon_schedule()
        # version user data
        self.num_samples_seen = 0

    def create_q_networks(self, representation_net):
        network = representation_net(self._num_input)
        network_output_dim = self._num_output
        self.network = DQNNetwork(network, network_output_dim, self._num_output).to(
            self.device
        )
        self.network.apply(self._init_fn)
        self.target_network = copy.deepcopy(self.network).requires_grad_(False)

    def get_model_user_data(self):
        return {
            "environment_implementation": self._environment_implementation,
            "num_input": self._num_input,
            "num_output": self._num_output,
        }

    def save(self, model_data_f):
        torch.save((self.network.state_dict(), self._epsilon_schedule), model_data_f)
        return {"num_samples_seen": self.num_samples_seen}

    @classmethod
    def load(cls, model_id, version_number, model_user_data, version_user_data, model_data_f):
        # Create the model instance
        model = HiveDQNModel(
            model_id=model_id,
            version_number=version_number,
            environment_implementation=model_user_data["environment_implementation"],
            num_input=int(model_user_data["num_input"]),
            num_output=int(model_user_data["num_output"]),
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
            HiveDQNModel, config.model_id, config.model_version
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
                        HiveDQNModel, config.model_id, config.model_version
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


class HiveDQNTraining:
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

        model = HiveDQNModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=flattened_dimensions(self._environment_specs.observation_space),
            num_output=flattened_dimensions(self._environment_specs.action_space),
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
