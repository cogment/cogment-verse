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

############ TUTORIAL STEP 2 ############
from cogment_verse_torch_agents.utils.tensors import tensor_from_cog_action, tensor_from_cog_obs
from data_pb2 import (
    ActorParams,
    AgentAction,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentSpecs,
    HumanConfig,
    HumanRole,
    SimpleBCTrainingRunConfig,
    TrialConfig,
)

##########################################


log = logging.getLogger(__name__)

# pylint: disable=arguments-differ


class SimpleBCAgentAdapterTutorialStep2(AgentAdapter):
    def __init__(self):
        super().__init__()
        self._dtype = torch.float

    @staticmethod
    async def run_async(func, *args):
        """Run a given function asynchronously in the default thread pool"""
        event_loop = asyncio.get_running_loop()
        return await event_loop.run_in_executor(None, func, *args)

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

                ############ TUTORIAL STEP 2 ############
                observation = tensor_from_cog_obs(sample.get_actor_observation(0), dtype=self._dtype)

                # The actor order matches the order in the trial configuration
                agent_action = sample.get_actor_action(0)
                teacher_action = sample.get_actor_action(1)

                # Check for teacher override.
                # Teacher action -1 corresponds to teacher approval,
                # i.e. the teacher considers the action taken by the agent to be correct
                if teacher_action.discrete_action != -1:
                    action = tensor_from_cog_action(teacher_action)
                    run_sample_producer_session.produce_training_sample((True, observation, action))
                else:
                    action = tensor_from_cog_action(agent_action)
                    run_sample_producer_session.produce_training_sample((False, observation, action))
                ##########################################

        async def run_impl(run_session):
            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

            config = run_session.config
            assert config.environment.specs.num_players == 1

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
                    human_config=HumanConfig(
                        environment_specs=env_params.specs,
                        role=HumanRole.TEACHER,
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
                        config=EnvironmentConfig(seed=12, framestack=1, render=True, render_width=256),
                    )
                ),
            )
        }
