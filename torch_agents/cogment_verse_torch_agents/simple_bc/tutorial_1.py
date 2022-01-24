# Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import asyncio
import copy
import logging

import cogment
import numpy as np
import torch
from cogment.api.common_pb2 import TrialState
from cogment_verse import AgentAdapter, MlflowExperimentTracker
from data_pb2 import (
    ActorParams,
    AgentAction,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentSpecs,
    SimpleBCTrainingRunConfig,
    TeacherConfig,
    TrialConfig,
)

log = logging.getLogger(__name__)

# pylint: disable=arguments-differ


class SimpleBCAgentAdapterTutorialStep1(AgentAdapter):
    def __init__(self):
        super().__init__()
        self._dtype = torch.float

    @staticmethod
    async def run_async(func, *args):
        """Run a given function asynchronously in the default thread pool"""
        event_loop = asyncio.get_running_loop()
        return await event_loop.run_in_executor(None, func, *args)

    def _create(
        self,
        model_id,
        **kwargs,
    ):
        raise NotImplementedError

    def _load(self, model_id, version_number, version_user_data, model_data_f):
        raise NotImplementedError

    def _save(self, model, model_data_f):
        raise NotImplementedError

    def _create_actor_implementations(self):
        async def impl(actor_session):
            actor_session.start()

            config = actor_session.config

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    action = np.random.default_rng().integers(0, config.environment_specs.num_action)
                    actor_session.do_action(AgentAction(discrete_action=action))

        return {
            "simple_bc": (impl, ["agent"]),
        }

    def _create_run_implementations(self):
        async def sample_producer_impl(run_sample_producer_session):
            assert run_sample_producer_session.count_actors() == 2

            async for sample in run_sample_producer_session.get_all_samples():
                if sample.get_trial_state() == TrialState.ENDED:
                    break

                log.info("Got raw sample")

        async def run_impl(run_session):
            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

            config = run_session.config
            assert config.environment.config.player_count == 1

            xp_tracker.log_params(
                config.training,
                config.environment.config,
                environment=config.environment.specs.implementation,
                policy_network_hidden_size=config.policy_network.hidden_size,
            )

            # Helper function to create a trial configuration
            def create_trial_config(trial_idx):
                env_params = copy.deepcopy(config.environment)
                env_params.config.seed = env_params.config.seed + trial_idx
                agent_actor_params = ActorParams(
                    name="agent_1",
                    actor_class="agent",
                    implementation="simple_bc",
                    agent_config=AgentConfig(
                        run_id=run_session.run_id,
                        environment_specs=env_params.specs,
                    ),
                )

                teacher_actor_params = ActorParams(
                    name="web_actor",
                    actor_class="teacher_agent",
                    implementation="client",
                    teacher_config=TeacherConfig(
                        environment_specs=env_params.specs,
                    ),
                )

                return TrialConfig(
                    run_id=run_session.run_id,
                    environment=env_params,
                    actors=[agent_actor_params, teacher_actor_params],
                )

            # Rollout a bunch of trials
            async for (
                _step_idx,
                _step_timestamp,
                _trial_id,
                _tick_id,
                sample,
            ) in run_session.start_trials_and_wait_for_termination(
                trial_configs=[create_trial_config(trial_idx) for trial_idx in range(config.training.trial_count)],
                max_parallel_trials=config.training.max_parallel_trials,
            ):
                log.info(f"Got sample {sample}")

        return {
            "simple_bc_training": (
                sample_producer_impl,
                run_impl,
                SimpleBCTrainingRunConfig(
                    environment=EnvironmentParams(
                        specs=EnvironmentSpecs(implementation="gym/LunarLander-v2", num_input=8, num_action=4),
                        config=EnvironmentConfig(seed=12, player_count=1, framestack=1, render=True, render_width=256),
                    )
                ),
            )
        }
