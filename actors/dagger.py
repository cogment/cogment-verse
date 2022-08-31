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

import cogment
import torch

import numpy as np

from cogment_verse import Model
from cogment_verse.specs import (
    AgentConfig,
    cog_settings,
    EnvironmentConfig,
    flatten,
    flattened_dimensions,
    PLAYER_ACTOR_CLASS,
    PlayerAction,
    SpaceValue,
    TEACHER_ACTOR_CLASS,
)
from actors.simple_a2c import SimpleA2CModel

torch.multiprocessing.set_sharing_strategy("file_system")

log = logging.getLogger(__name__)

class LearnerModel(Model):
    def __init__(
        self,
        model_id,
        environment_implementation,
        num_input,
        num_output,
        policy_network_num_hidden_nodes=64,
        version_number=0,
    ):
        super().__init__(model_id, version_number)

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
            "environment_implementation": self._environment_implementation,
            "num_input": self._num_input,
            "num_output": self._num_output,
            "policy_network_num_hidden_nodes": self._policy_network_num_hidden_nodes,
        }

    def save(self, model_data_f):
        torch.save(self.policy_network.state_dict(), model_data_f)

        return {"total_samples": self.total_samples}

    @classmethod
    def load(cls, model_id, version_number, model_user_data, version_user_data, model_data_f):
        # Create the model instance
        model = LearnerModel(
            model_id=model_id,
            version_number=version_number,
            environment_implementation=model_user_data["environment_implementation"],
            num_input=int(model_user_data["num_input"]),
            num_output=int(model_user_data["num_output"]),
            policy_network_num_hidden_nodes=int(model_user_data["policy_network_num_hidden_nodes"]),
        )

        # Load the saved states
        policy_network_state_dict = torch.load(model_data_f)
        model.policy_network.load_state_dict(policy_network_state_dict)

        # Load version data
        model.total_samples = version_user_data["total_samples"]
        return model

class DaggerTeacher:
    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        return [TEACHER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()
        config = actor_session.config
        observation_space = config.environment_specs.observation_space
        model, _, _ = await actor_session.model_registry.retrieve_version(
            SimpleA2CModel, config.model_id, -1
        )

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation_tensor = torch.tensor(
                    flatten(observation_space, event.observation.observation.value), dtype=self._dtype
                )
                probs = torch.softmax(model.actor_network(observation_tensor), dim=-1)
                discrete_action_tensor = torch.distributions.Categorical(probs).sample()
                action_value = SpaceValue(properties=[SpaceValue.PropertyValue(discrete=discrete_action_tensor.item())])
                actor_session.do_action(PlayerAction(value=action_value))


class DaggerLearner:
    def __init__(self, _cfg):
        self._dtype = torch.float

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()
        config = actor_session.config
        observation_space = config.environment_specs.observation_space
        model, _model_info, version_info = await actor_session.model_registry.retrieve_version(
            LearnerModel, config.model_id, config.model_version
        )
        model.policy_network.eval()

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                observation_tensor = torch.tensor(
                    flatten(observation_space, event.observation.observation.value), dtype=self._dtype
                )
                scores = model.policy_network(observation_tensor.view(1, -1))
                probs = torch.softmax(scores, dim=-1)
                discrete_action_tensor = torch.distributions.Categorical(probs).sample()
                action_value = SpaceValue(properties=[SpaceValue.PropertyValue(discrete=discrete_action_tensor.item())])
                actor_session.do_action(PlayerAction(value=action_value))

class DaggerTraining:
    default_cfg = {
        "seed": 12,
        "num_trials": 4,
        "start_learning_trial": 2,
        "discount_factor": 0.95,
        "learning_rate": 0.01,
        "batch_size": 32,
        "policy_network": {"num_hidden_nodes": 64},
    }

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._dtype = torch.float
        self._environment_specs = environment_specs
        self._cfg = cfg

    async def sample_producer(self, sample_producer_session):
        assert len(sample_producer_session.trial_info.parameters.actors) == 2
        assert self._cfg.teacher_model == "SimpleA2CModel"
        assert self._environment_specs.action_space.properties[0].WhichOneof("type") == "discrete"

        teachers_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == TEACHER_ACTOR_CLASS
        ]
        learner_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == PLAYER_ACTOR_CLASS
        ]
        
        assert len(learner_params) == 1
        assert len(teachers_params) == 1
        
        teacher_params = teachers_params[0]
        learner_params = learner_params[0]

        environment_specs = teacher_params.config.environment_specs

        async for sample in sample_producer_session.all_trial_samples():
            teacher_action = sample.actors_data[teacher_params.name].action
            teacher_action_tensor = torch.tensor(
                flatten(environment_specs.action_space, teacher_action.value), dtype=self._dtype
            )
            learner_action = sample.actors_data[learner_params.name].action
            learner_action_tensor = torch.tensor(
                flatten(environment_specs.action_space, learner_action.value), dtype=self._dtype
            )
            observation_tensor = torch.tensor(
                flatten(environment_specs.observation_space, sample.actors_data[learner_params.name].observation.value),
                dtype=self._dtype,
            )
            sample_producer_session.produce_sample((observation_tensor, teacher_action_tensor, learner_action_tensor))

    async def impl(self, run_session):
        model_id = f"{run_session.run_id}_model"

        # Initializing a model
        learner_model = LearnerModel(
            model_id,
            environment_implementation=self._environment_specs.implementation,
            num_input=flattened_dimensions(self._environment_specs.observation_space),
            num_output=flattened_dimensions(self._environment_specs.action_space),
            policy_network_num_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
        )
        _model_info, _version_info = await run_session.model_registry.publish_initial_version(learner_model)

        run_session.log_params(
            self._cfg,
            environment_implementation=self._environment_specs.implementation,
            policy_network_num_hidden_nodes=self._cfg.policy_network.num_hidden_nodes,
        )

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx):
            
            player_actor_params = cogment.ActorParameters(
                cog_settings,
                name="player",
                class_name=PLAYER_ACTOR_CLASS,
                implementation="actors.dagger.DaggerLearner",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs,
                    model_id=model_id,
                    model_version=-1,
                ),
            )
            
            teacher_actor_params = cogment.ActorParameters(
                cog_settings,
                name="teacher",
                class_name=TEACHER_ACTOR_CLASS,
                implementation="actors.dagger.DaggerTeacher",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs,
                    model_id=self._cfg.teacher_model_id,
                ),
            )

            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id, render=False, seed=self._cfg.seed + trial_idx
                ),
                actors=[teacher_actor_params, player_actor_params],
            )

        # Configure the optimizer and loss function
        optimizer = torch.optim.Adam(
            learner_model.policy_network.parameters(),
            lr=self._cfg.learning_rate,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        # Store the accumulated observations/actions
        observations = []
        teacher_actions = []
        learner_actions = []

        teacher_model, _, _ = await run_session.model_registry.retrieve_version(
            SimpleA2CModel, self._cfg.teacher_model_id, -1
        )
        
        # Rollout a bunch of trials
        for (step_idx, _trial_id, _trial_idx, sample,) in run_session.start_and_await_trials(
            trials_id_and_params=[
                (f"{run_session.run_id}_{trial_idx}", create_trial_params(trial_idx))
                for trial_idx in range(self._cfg.num_trials)
            ],
            sample_producer_impl=self.sample_producer,
            num_parallel_trials=1,
        ):
            (observation, teacher_action, learner_action) = sample
            observations.append(observation)
            teacher_actions.append(teacher_action)
            learner_actions.append(learner_action)

            if len(observations) < self._cfg.batch_size:
                continue

            if _trial_idx >= self._cfg.start_learning_trial:
                # Feed the learner's observation to the teacher model to find the correct action
                probs = torch.softmax(teacher_model.actor_network(observation), dim=-1)
                discrete_action_tensor = torch.distributions.Categorical(probs).sample()
                action_value = SpaceValue(properties=[SpaceValue.PropertyValue(discrete=discrete_action_tensor.item())])
                correct_action = PlayerAction(value=action_value)
                correct_action_tensor = torch.tensor(
                    flatten(self._environment_specs.action_space, correct_action.value), dtype=self._dtype
                )
                teacher_actions[-1] = correct_action_tensor
            
            
            # Sample a batch of observations/actions
            batch_indices = np.random.default_rng().integers(0, len(observations), self._cfg.batch_size)
            batch_observation = torch.vstack([observations[i] for i in batch_indices])
            batch_action = torch.vstack([teacher_actions[i] for i in batch_indices])

            learner_model.policy_network.train()
            pred_policy = learner_model.policy_network(batch_observation)
            loss = loss_fn(pred_policy, batch_action)

            # Backprop!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Publish the newly trained version every 100 steps
            if step_idx % 100 == 0:
                version_info = await run_session.model_registry.publish_version(learner_model)

                run_session.log_metrics(
                    model_version_number=version_info["version_number"],
                    loss=loss.item(),
                    total_samples=len(observations),
                )
                
                
                
           
            
            
            
            
