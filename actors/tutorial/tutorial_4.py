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

import io
import logging

import cogment

############ TUTORIAL STEP 4 ############
import numpy as np
import torch
from gym.spaces import utils

from cogment_verse import Model
from cogment_verse.specs import (
    HUMAN_ACTOR_IMPL,
    PLAYER_ACTOR_CLASS,
    TEACHER_ACTOR_CLASS,
    WEB_ACTOR_NAME,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentSpecs,
    cog_settings,
)

#########################################


torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)


class SimpleBCModel(Model):
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
    def deserialize_model(cls, serialized_model) -> SimpleBCModel:
        stream = io.BytesIO(serialized_model)
        (policy_network_state_dict, model_user_data) = torch.load(stream)

        model = SimpleBCModel(
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


class SimpleBCActor:
    def __init__(self, _cfg):
        super().__init__()
        self._dtype = torch.float

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config

        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        observation_space = environment_specs.get_observation_space()
        action_space = environment_specs.get_action_space(seed=config.seed)

        # Get model
        if config.model_iteration == -1:
            latest_model = await actor_session.model_registry.track_latest_model(
                name=config.model_id, deserialize_func=SimpleBCModel.deserialize_model
            )
            model, _ = await latest_model.get()
        else:
            serialized_model = await actor_session.model_registry.retrieve_model(
                config.model_id, config.model_iteration
            )
            model = SimpleBCModel.deserialize_model(serialized_model)

        log.info(f"Starting trial with model v{model.iteration}")

        model.policy_network.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation = observation_space.deserialize(event.observation.observation)
                observation_tensor = torch.tensor(observation.flat_value, dtype=self._dtype)
                scores = model.policy_network(observation_tensor.view(1, -1))
                probs = torch.softmax(scores, dim=-1)
                discrete_action_tensor = torch.distributions.Categorical(probs).sample()
                action = action_space.create(value=discrete_action_tensor.item())
                actor_session.do_action(action_space.serialize(action))


class SimpleBCTraining:
    default_cfg = {
        "seed": 12,
        "num_trials": 10,
        ############ TUTORIAL STEP 4 ############
        "discount_factor": 0.95,
        "learning_rate": 0.01,
        "batch_size": 32,
        "train_only_from_demonstration": False,
        #########################################
        "policy_network": {"num_hidden_nodes": 64},
    }

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg

    async def sample_producer(self, sample_producer_session):
        assert len(sample_producer_session.trial_info.parameters.actors) == 2

        players_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == PLAYER_ACTOR_CLASS
        ]
        teachers_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == TEACHER_ACTOR_CLASS
        ]
        assert len(players_params) == 1
        assert len(teachers_params) == 1
        player_params = players_params[0]
        teacher_params = teachers_params[0]

        environment_specs = EnvironmentSpecs.deserialize(player_params.config.environment_specs)
        action_space = environment_specs.get_action_space()
        observation_space = environment_specs.get_observation_space()

        async for sample in sample_producer_session.all_trial_samples():
            observation_tensor = torch.tensor(
                observation_space.deserialize(sample.actors_data[player_params.name].observation).flat_value,
                dtype=self._dtype,
            )

            teacher_action = action_space.deserialize(sample.actors_data[teacher_params.name].action)

            if teacher_action.flat_value is not None:
                applied_action = teacher_action
                demonstration = True
            else:
                applied_action = action_space.deserialize(sample.actors_data[player_params.name].action)
                demonstration = False

            if applied_action.flat_value is None:
                # TODO figure out why we get into this situation
                continue

            action_tensor = torch.tensor(applied_action.flat_value, dtype=self._dtype)
            sample_producer_session.produce_sample((demonstration, observation_tensor, action_tensor))

    async def impl(self, run_session):
        assert self._environment_specs.num_players == 1

        model_id = f"{run_session.run_id}_model"

        # Initializing a model
        model = SimpleBCModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=utils.flatdim(self._environment_specs.get_observation_space().gym_space),
            num_output=utils.flatdim(self._environment_specs.get_action_space().gym_space),
            policy_network_num_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
        )
        serialized_model = SimpleBCModel.serialize_model(model)
        iteration_info = await run_session.model_registry.publish_model(
            name=model_id,
            model=serialized_model,
        )

        run_session.log_params(
            self._cfg,
            environment_implementation=self._environment_specs.implementation,
            policy_network_num_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
        )

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx):
            agent_actor_params = cogment.ActorParameters(
                cog_settings,
                name="player",
                class_name=PLAYER_ACTOR_CLASS,
                ############ TUTORIAL STEP 4 ############
                implementation="actors.tutorial.tutorial_4.SimpleBCActor",
                #########################################
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                    model_id=model_id,
                    model_iteration=-1,
                ),
            )

            teacher_actor_params = cogment.ActorParameters(
                cog_settings,
                name=WEB_ACTOR_NAME,
                class_name=TEACHER_ACTOR_CLASS,
                implementation=HUMAN_ACTOR_IMPL,
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                ),
            )

            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id, render=True, seed=self._cfg.seed + trial_idx
                ),
                actors=[agent_actor_params, teacher_actor_params],
            )

        ############ TUTORIAL STEP 4 ############
        # Configure the optimizer
        optimizer = torch.optim.Adam(
            model.policy_network.parameters(),
            lr=self._cfg.learning_rate,
        )

        # Keep accumulated observations/actions around
        observations = []
        actions = []

        loss_fn = torch.nn.CrossEntropyLoss()
        ##########################################

        # Rollout a bunch of trials
        for (step_idx, _trial_id, _trial_idx, sample,) in run_session.start_and_await_trials(
            trials_id_and_params=[
                (f"{run_session.run_id}_{trial_idx}", create_trial_params(trial_idx))
                for trial_idx in range(self._cfg.num_trials)
            ],
            sample_producer_impl=self.sample_producer,
            num_parallel_trials=1,
        ):
            ############ TUTORIAL STEP 4 ############
            (demonstration, observation, action) = sample
            if self._cfg.train_only_from_demonstration and not demonstration:
                continue

            observations.append(observation)
            actions.append(action)

            if len(observations) < self._cfg.batch_size:
                continue

            # Sample a batch of observations/actions
            batch_indices = np.random.default_rng().integers(0, len(observations), self._cfg.batch_size)
            batch_obs = torch.vstack([observations[i] for i in batch_indices])
            batch_act = torch.vstack([actions[i] for i in batch_indices])

            model.policy_network.train()
            pred_policy = model.policy_network(batch_obs)
            loss = loss_fn(pred_policy, batch_act)

            # Backprop!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Publish the newly trained version every 100 steps
            if step_idx % 100 == 0:
                serialized_model = SimpleBCModel.serialize_model(model)
                iteration_info = await run_session.model_registry.store_model(
                    name=model_id,
                    model=serialized_model,
                )
                run_session.log_metrics(
                    model_iteration=iteration_info.iteration,
                    loss=loss.item(),
                    total_samples=len(observations),
                )
            ##########################################
