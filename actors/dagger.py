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
                obs_tensor = torch.tensor(
                    flatten(observation_space, event.observation.observation.value), dtype=self._dtype
                )
                probs = torch.softmax(model.actor_network(obs_tensor), dim=-1)
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

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                actor_session.do_action(PlayerAction())


class DaggerTraining:
    default_cfg = {
        "seed": 12,
        "num_trials": 1,
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

        teachers_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == TEACHER_ACTOR_CLASS
        ]
        teacher_params = teachers_params[0]

        learner_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == PLAYER_ACTOR_CLASS
        ]
        learner_params = learner_params[0]

        environment_specs = teacher_params.config.environment_specs

        async for sample in sample_producer_session.all_trial_samples():
            
            sample_producer_session.produce_sample((None, None, None))

    async def impl(self, run_session):

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
                    run_id=run_session.run_id, render=True, seed=self._cfg.seed + trial_idx
                ),
                actors=[teacher_actor_params, player_actor_params],
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
            pass