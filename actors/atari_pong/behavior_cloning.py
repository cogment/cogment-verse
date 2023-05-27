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

import gc
import io
import logging
from enum import Enum

import cogment
import cv2
import numpy as np
import torch
from gym.spaces import utils

from cogment_verse import Model
from cogment_verse.specs import ActorClass, EnvironmentSpecs

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)


def check_gc_tensor():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def count_gc_tensor() -> int:
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                count += 1
        except:
            pass
    return count


def resize_frame(frame):
    frame = frame[14:-5,5:-4]
    frame = cv2.resize(frame, (84,84), interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame, dtype = np.uint8)
    return frame


def resize_obs(observation: np.ndarray) -> np.ndarray:
    new_obs = observation.copy()[:, :, :4]
    for i in range(new_obs.shape[2]):
        new_obs[:, :, i] = resize_frame(new_obs[:, :, i])

    return new_obs


class DataAugmentationEnum(Enum):
    SINGLE_PLAYER = "single_player"
    ALL_PLAYERS = "all_players"
    FLIP = "flip"


class PongCNNModel(Model):
    def __init__(
        self,
        model_id,
        environment_implementation,
        input_shape,
        num_output,
        network_num_hidden_nodes=64,
        dtype=torch.float32,
        device: str = "cpu",
        iteration=0,
    ):
        super().__init__(model_id, iteration)

        self._dtype = dtype
        self._environment_implementation = environment_implementation
        self._input_shape = input_shape
        self._num_output = num_output
        self._network_num_hidden_nodes = network_num_hidden_nodes

        self.device = device
        # self.network = torch.nn.Sequential(
        #     torch.nn.Conv2d(num_input, network_num_hidden_nodes, dtype=self._dtype),
        #     torch.nn.BatchNorm1d(network_num_hidden_nodes, dtype=self._dtype),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(network_num_hidden_nodes, network_num_hidden_nodes, dtype=self._dtype),
        #     torch.nn.BatchNorm1d(network_num_hidden_nodes, dtype=self._dtype),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(network_num_hidden_nodes, num_output, dtype=self._dtype),
        # )

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[2], network_num_hidden_nodes, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(network_num_hidden_nodes, 2*network_num_hidden_nodes, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2*network_num_hidden_nodes, 2*network_num_hidden_nodes, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_output)
        )

        self.network.to(torch.device(self.device))
        self.total_samples = 0

    def _calculate_flatten_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        flattened_size = self.network[:6](dummy_input).shape[1]
        return flattened_size

    def get_model_user_data(self):
        return {
            "model_id": self.model_id,
            "iteration": self.iteration,
            "environment_implementation": self._environment_implementation,
            "input_shape": self._input_shape,
            "num_output": self._num_output,
            "network_num_hidden_nodes": self._network_num_hidden_nodes,
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
    def deserialize_model(cls, serialized_model) -> PongCNNModel:
        stream = io.BytesIO(serialized_model)
        (network_state_dict, model_user_data) = torch.load(stream)

        model = PongCNNModel(
            model_id=model_user_data["model_id"],
            iteration=model_user_data["iteration"],
            environment_implementation=model_user_data["environment_implementation"],
            input_shape=model_user_data["input_shape"],
            num_output=int(model_user_data["num_output"]),
            network_num_hidden_nodes=int(model_user_data["network_num_hidden_nodes"]),
        )
        model.network.load_state_dict(network_state_dict)
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
                name=config.model_id, deserialize_func=PongCNNModel.deserialize_model
            )
            model, _ = await latest_model.get()
        else:
            serialized_model = await actor_session.model_registry.retrieve_model(
                config.model_id, config.model_iteration
            )
            model = PongCNNModel.deserialize_model(serialized_model)

        log.info(f"Starting trial with model_id: {model.model_id} | iteration: {model.iteration}")

        model.network.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation = observation_space.deserialize(event.observation.observation)
                if observation.current_player is not None and observation.current_player != actor_session.name:
                    # Not the turn of the agent
                    actor_session.do_action(action_space.serialize(action_space.create()))
                    continue

                resized_obs = resize_obs(observation.value)

                observation_tensor = torch.tensor(resized_obs, dtype=self._dtype)
                observation_tensor = torch.unsqueeze(observation_tensor.permute((2, 0, 1)), dim=0).clone()

                with torch.no_grad():
                    scores = model.network(observation_tensor)
                    probs = torch.softmax(scores, dim=-1)
                action_value = torch.distributions.Categorical(probs).sample().cpu().item()
                action = action_space.create(value=action_value)
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
        self.model = PongCNNModel(
            model_id="",
            environment_implementation=self._environment_specs.implementation,
            input_shape=(84, 84, 4),
            num_output=utils.flatdim(self._environment_specs.get_action_space().gym_space),
            network_num_hidden_nodes=self._cfg.network.num_hidden_nodes,
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

        log.debug(f"{torch.cuda.memory_allocated()} | Start memory allocated")

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

                resized_obs = resize_obs(player_observation.value)

                actions.append(torch.tensor(player_action.flat_value, dtype=self._dtype).detach())
                demonstrations.append(False)  # No teacher in our current scenario
                observations.append(torch.tensor(resized_obs, dtype=self._dtype).detach())
                rewards.append(torch.tensor(player_reward if player_reward is not None else 0, dtype=self._dtype).detach())

                log.debug(f"{torch.cuda.memory_allocated()} | After appending a sample to list")

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
                    actions.append(torch.tensor(player_action.flat_value, dtype=self._dtype).detach())
                    demonstrations.append(False)  # No teacher in our current scenario
                    observations.append(torch.tensor(obs.flat_value, dtype=self._dtype).detach())
                    rewards.append(torch.tensor(player_reward if player_reward is not None else 0, dtype=self._dtype).detach())

            for action, demonstration, observation, reward in zip(actions, demonstrations, observations, rewards):
                sample_producer_session.produce_sample((demonstration, observation, action, reward))

            log.debug(f"{torch.cuda.memory_allocated()} | After producing all samples")

    async def impl(self, run_session):
        # assert self._environment_specs.num_players == 1

        # global gc_tensor_count
        # gc_tensor_count = count_gc_tensor()


        if self._cfg.model_id:
            model_id = self._cfg.model_id
        else:
            model_id = f"{run_session.run_id}_model"
            self._cfg.model_id = model_id
        self.model.model_id = model_id

        serialized_model = PongCNNModel.serialize_model(self.model)
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
            self.model.network.parameters(),
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
                observations.append(trial_observation.clone())
                actions.append(trial_action.clone())
                rewards.append(trial_reward.clone())

                if len(observations) < self._cfg.batch_size:
                    continue

                # Sample a batch of observations/actions
                batch_indices = np.random.default_rng().integers(0, len(observations), self._cfg.batch_size)
                batch_obs = torch.vstack([torch.unsqueeze(observations[i], axis=0) for i in batch_indices]).permute(0, 3, 1, 2).detach()
                batch_act = torch.vstack([actions[i] for i in batch_indices]).detach()

                log.debug(f"{torch.cuda.memory_allocated()} | before training")

                self.model.network.train()
                # print(f"batch obs shape: {batch_obs.size()}")
                # print(f"batch act shape: {batch_act.size()}")
                pred_policy = self.model.network(batch_obs)
                loss = loss_fn(pred_policy, batch_act)

                log.debug(f"{torch.cuda.memory_allocated()} | After training")

                model_updates += 1

                # Backprop!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Publish the newly trained iteration every 100 steps
                if step_idx % self._cfg.update_frequency == 0:
                    serialized_model = PongCNNModel.serialize_model(self.model)
                    iteration_info = await run_session.model_registry.store_model(
                        name=model_id,
                        model=serialized_model,
                    )

                    run_session.log_metrics(
                        model_iteration=iteration_info.iteration,
                        loss=loss.item(),
                        total_samples=len(observations),
                    )


                # print(f"GC Tensors: {gc_tensor_count - count_gc_tensor()}")
                # gc_tensor_count = count_gc_tensor()

            log.info(f"Epoch {epoch_idx+1} completed with {len(observations)} samples, {model_updates} model updates.")
