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
from enum import Enum

import io
import logging

import cogment
import numpy as np
import torch
from gym.spaces import utils

from cogment_verse import Model
from cogment_verse.specs import ActorClass, EnvironmentSpecs

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)


class DataAugmentationEnum(Enum):
    SINGLE_PLAYER = "single_player"
    ALL_PLAYERS = "all_players"
    FLIP = "flip"


class BehaviorCloningModel(Model):
    def __init__(
        self,
        model_id,
        environment_implementation,
        num_input,
        num_output,
        policy_network_num_hidden_nodes=64,
        iteration=0,
    ):
        super().__init__(model_id, iteration)

        self._dtype = torch.float
        self._environment_implementation = environment_implementation
        self._num_input = num_input
        self._num_output = num_output
        self._policy_network_num_hidden_nodes = policy_network_num_hidden_nodes

        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(num_input, policy_network_num_hidden_nodes, dtype=self._dtype),
            torch.nn.BatchNorm1d(policy_network_num_hidden_nodes, dtype=self._dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(policy_network_num_hidden_nodes, policy_network_num_hidden_nodes, dtype=self._dtype),
            torch.nn.BatchNorm1d(policy_network_num_hidden_nodes, dtype=self._dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(policy_network_num_hidden_nodes, num_output, dtype=self._dtype),
        )

        self.total_samples = 0

    def get_model_user_data(self):
        return {
            "model_id": self.model_id,
            "iteration": self.iteration,
            "environment_implementation": self._environment_implementation,
            "num_input": self._num_input,
            "num_output": self._num_output,
            "policy_network_num_hidden_nodes": self._policy_network_num_hidden_nodes,
            "total_samples": self.total_samples,
        }

    @staticmethod
    def serialize_model(model) -> bytes:
        stream = io.BytesIO()
        torch.save(
            (
                model.policy_network.state_dict(),
                model.get_model_user_data(),
            ),
            stream,
        )
        return stream.getvalue()

    @classmethod
    def deserialize_model(cls, serialized_model) -> BehaviorCloningModel:
        stream = io.BytesIO(serialized_model)
        (policy_network_state_dict, model_user_data) = torch.load(stream)

        model = BehaviorCloningModel(
            model_id=model_user_data["model_id"],
            iteration=model_user_data["iteration"],
            environment_implementation=model_user_data["environment_implementation"],
            num_input=int(model_user_data["num_input"]),
            num_output=int(model_user_data["num_output"]),
            policy_network_num_hidden_nodes=int(model_user_data["policy_network_num_hidden_nodes"]),
        )
        model.policy_network.load_state_dict(policy_network_state_dict)
        model.total_samples = model_user_data["total_samples"]
        return model


class BehaviorCloningActor:
    def __init__(self, _cfg):
        super().__init__()
        self._dtype = torch.float

    def get_actor_classes(self):
        return [ActorClass.PLAYER.value]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config
        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        observation_space = environment_specs.get_observation_space()
        action_space = environment_specs.get_action_space(seed=config.seed)

        # Get model
        if config.model_iteration == -1:
            latest_model = await actor_session.model_registry.track_latest_model(
                name=config.model_id, deserialize_func=BehaviorCloningModel.deserialize_model
            )
            model, _ = await latest_model.get()
        else:
            serialized_model = await actor_session.model_registry.retrieve_model(
                config.model_id, config.model_iteration
            )
            model = BehaviorCloningModel.deserialize_model(serialized_model)

        log.info(f"Starting trial with model_id: {model.model_id} | iteration: {model.iteration}")

        model.policy_network.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation = observation_space.deserialize(event.observation.observation)

                observation_tensor = torch.tensor(observation.flat_value, dtype=self._dtype).view(1, -1)

                with torch.no_grad():
                    scores = model.policy_network(observation_tensor)
                    probs = torch.softmax(scores, dim=-1)
                discrete_action_tensor = torch.distributions.Categorical(probs).sample()
                action = action_space.create(value=discrete_action_tensor.item())
                actor_session.do_action(action_space.serialize(action))


class BehaviorCloningTrainingOffline:
    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg
        self._data_augmentation_type = DataAugmentationEnum(cfg.data_augmentation.type)
        self._data_augmentation_cfg = cfg.data_augmentation


        # Initializing a model
        self.model = BehaviorCloningModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            num_input=utils.flatdim(self._environment_specs.get_observation_space().gym_space),
            num_output=utils.flatdim(self._environment_specs.get_action_space().gym_space),
            policy_network_num_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
        )

    async def sample_producer(self, sample_producer_session):

        players_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == ActorClass.PLAYER.value
        ]

        player_names = [params.name for params in players_params]

        player_params = players_params[0]
        environment_specs = EnvironmentSpecs.deserialize(player_params.config.environment_specs)
        action_space = environment_specs.get_action_space()
        observation_space = environment_specs.get_observation_space()

        async for sample in sample_producer_session.all_trial_samples():

            actions = []
            demonstrations = []
            observations = []
            rewards = []

            if self._data_augmentation_type == DataAugmentationEnum.SINGLE_PLAYER:
                assert "player" in self._data_augmentation_cfg, "The 'player' parameter is missing from the 'data_augmentation' parameters."
                assert self._data_augmentation_cfg.player in player_names, f"The 'player' parameter does not match with any player actor present in the trial: [{', '.join(player_names)}]."
                player_action = action_space.deserialize(sample.actors_data[self._data_augmentation_cfg.player].action)
                player_observation = observation_space.deserialize(sample.actors_data[self._data_augmentation_cfg.player].observation)
                player_reward = sample.actors_data[self._data_augmentation_cfg.player].reward

                if player_action.flat_value is None:
                    continue

                actions.append(torch.tensor(player_action.flat_value, dtype=self._dtype))
                demonstrations.append(False)  # No teacher in our current scenario
                observations.append(torch.tensor(player_observation.flat_value, dtype=self._dtype))
                rewards.append(torch.tensor(player_reward if player_reward is not None else 0, dtype=self._dtype))

            elif self._data_augmentation_type == DataAugmentationEnum.ALL_PLAYERS:
                for player in players_params:
                    player_action = action_space.deserialize(sample.actors_data[player.name].action)
                    player_observation = observation_space.deserialize(sample.actors_data[player.name].observation)
                    player_reward = sample.actors_data[player.name].reward

                    if player_action.flat_value is None:
                        continue

                    actions.append(torch.tensor(player_action.flat_value, dtype=self._dtype))
                    demonstrations.append(False)  # No teacher in our current scenario
                    observations.append(torch.tensor(player_observation.flat_value, dtype=self._dtype))
                    rewards.append(torch.tensor(player_reward if player_reward is not None else 0, dtype=self._dtype))

            elif self._data_augmentation_type == DataAugmentationEnum.FLIP:
                assert "player" in self._data_augmentation_cfg, "The 'player' parameter is missing from the 'data_augmentation' parameters."
                assert self._data_augmentation_cfg.player in [params.name for params in players_params], "The 'player' parameter does not match with any player actor present in the trial."
                player_action = action_space.deserialize(sample.actors_data[self._data_augmentation_cfg.player].action)
                player_observation = observation_space.deserialize(sample.actors_data[self._data_augmentation_cfg.player].observation)
                player_reward = sample.actors_data[self._data_augmentation_cfg.player].reward

                if player_action.flat_value is None:
                        continue


                observation_tensor = torch.tensor(player_observation.value, dtype=self._dtype)
                assert len(observation_tensor.size()) == 3, f"Observation tensor has {len(observation_tensor.size())} dimensions."
                assert observation_tensor.size() == torch.Size([84, 84, 6]), f"Observation tensor has size {observation_tensor.size()}."

                last_dim_idx = torch.arange(observation_tensor.size(-1))
                # flipped_observation_tensor = observation_tensor[:, :, last_dim_idx[0:4] + last_dim_idx[-1, -2]]
                flipped_observation_tensor = observation_tensor[:, :, [0, 1, 2, 3, 5, 4]]
                flipped_observation = observation_space.create(value=flipped_observation_tensor)

                for obs in [player_observation, flipped_observation]:
                    actions.append(torch.tensor(player_action.flat_value, dtype=self._dtype))
                    demonstrations.append(False)  # No teacher in our current scenario
                    observations.append(torch.tensor(obs.flat_value, dtype=self._dtype))
                    rewards.append(torch.tensor(player_reward if player_reward is not None else 0, dtype=self._dtype))

            for action, demonstration, observation, reward in zip(actions, demonstrations, observations, rewards):
                sample_producer_session.produce_sample((demonstration, observation, action, reward))

    async def impl(self, run_session):
        # assert self._environment_specs.num_players == 1

        if self._cfg.model_id:
            model_id = self._cfg.model_id
        else:
            model_id = f"{run_session.run_id}_model"
            self._cfg.model_id = model_id
        self.model.model_id = model_id

        serialized_model = BehaviorCloningModel.serialize_model(self.model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )

        run_session.log_params(
            self._cfg,
            environment_implementation=self._environment_specs.implementation,
        )

        # Configure the optimizer
        optimizer = torch.optim.Adam(
            self.model.policy_network.parameters(),
            lr=self._cfg.learning_rate,
        )

        loss_fn = torch.nn.CrossEntropyLoss()

        model_updates = 0
        # To repeat the training iteration over the dataset of trials.
        for epoch_idx in range(self._cfg.num_epochs):

            # Keep accumulated observations/actions around
            demonstrations = []
            observations = []
            actions = []
            rewards = []

            # One iteration per trial (all trial samples)
            for (step_idx, _trial_id, _trial_idx, sample,) in run_session.load_trials(
                sample_producer_impl=self.sample_producer,
                trial_ids=self._cfg.trial_ids,
                num_trials=self._cfg.num_trials,
            ):

                (trial_demonstration, trial_observation, trial_action, trial_reward) = sample

                demonstrations.append(trial_demonstration)
                observations.append(trial_observation)
                actions.append(trial_action)
                rewards.append(trial_reward)

                if len(observations) < self._cfg.batch_size:
                    continue

                # Sample a batch of observations/actions
                batch_indices = np.random.default_rng().integers(0, len(observations), self._cfg.batch_size)
                batch_obs = torch.vstack([observations[i] for i in batch_indices])
                batch_act = torch.vstack([actions[i] for i in batch_indices])

                self.model.policy_network.train()
                pred_policy = self.model.policy_network(batch_obs)
                loss = loss_fn(pred_policy, batch_act)

                model_updates += 1

                # Backprop!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Publish the newly trained iteration every 100 steps
                if step_idx % self._cfg.update_frequency == 0:
                    serialized_model = BehaviorCloningModel.serialize_model(self.model)
                    iteration_info = await run_session.model_registry.store_model(
                        name=model_id,
                        model=serialized_model,
                    )

                    run_session.log_metrics(
                        model_iteration=iteration_info.iteration,
                        loss=loss.item(),
                        total_samples=len(observations),
                    )

            log.info(f"Epoch {epoch_idx+1} completed with {len(observations)} samples, {model_updates} model updates.")
