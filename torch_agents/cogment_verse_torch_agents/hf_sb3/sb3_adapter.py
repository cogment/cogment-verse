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
from collections import namedtuple

import cogment

############ TUTORIAL STEP 4 ############
import numpy as np

##########################################
import torch
from cogment.api.common_pb2 import TrialState
from cogment_verse import AgentAdapter, MlflowExperimentTracker
from cogment_verse_torch_agents.utils.tensors import cog_action_from_tensor, tensor_from_cog_action, tensor_from_cog_obs
from data_pb2 import (
    ActorParams,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentSpecs,
    MLPNetworkConfig,
    TrialConfig,
    SimpleSB3TrainingRunConfig,
)
from huggingface_sb3 import load_from_hub, push_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# SimpleSB3Model = namedtuple("SimpleSB3Model", ["model_id", "version_number", "policy_network"])

log = logging.getLogger(__name__)

# pylint: disable=arguments-differ
class SimpleSB3AgentAdapter(AgentAdapter):
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
        environment_specs,
        policy_network_hidden_size=64,
        **kwargs,
    ):
        checkpoint = load_from_hub(
            repo_id="sb3/demo-hf-CartPole-v1",
            filename="ppo-CartPole-v1",
        )
        model = SimpleSB3Model(
            model_id = model_id,
            version_number = 1,
            policy_network = PPO.load(checkpoint)
        )

        model_user_data = {
            "environment_implementation": environment_specs.implementation,
            "num_input": environment_specs.num_input,
            "num_action": environment_specs.num_action,
        }

        return model, model_user_data


    def _create_actor_implementations(self):
        async def impl(actor_session):
            actor_session.start()

            config = actor_session.config

            model, _model_info, version_info = await self.retrieve_version(config.model_id, config.model_version)
            model_version_number = version_info["version_number"]
            log.info(f"Starting trial with model v{model_version_number}")

            # Retrieve the policy network and set it to "eval" mode
            policy_network = copy.deepcopy(model.policy_network)
            policy_network.eval()

            @torch.no_grad()
            def compute_action(event):
                obs = tensor_from_cog_obs(event.observation.snapshot, dtype=self._dtype)
                scores = policy_network(obs.view(1, -1))
                probs = torch.softmax(scores, dim=-1)
                action = torch.distributions.Categorical(probs).sample()
                return action

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    action = await self.run_async(compute_action, event)
                    actor_session.do_action(cog_action_from_tensor(action))

        return {
            "simple_bc": (impl, ["agent"]),
        }

    def _create_run_implementations(self):

        async def run_impl(run_session):
            # xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)
            config = run_session.config
            actors_params = [
                ActorParams(
                    name=actor_params.name,
                    actor_class=actor_params.actor_class,
                    implementation=actor_params.implementation,
                    agent_config=AgentConfig(
                        run_id=run_session.run_id,
                        environment_specs=config.environment.specs,
                        model_id=actor_params.agent_config.model_id,
                        model_version=actor_params.agent_config.model_version,
                    ),
                )
                for actor_params in config.actors[: config.environment.specs.num_players]
            ]

            config = run_session.config
            assert config.environment.specs.num_players == 1

            # xp_tracker.log_params(
            #     config.training,
            #     config.environment.config,
            #     environment=config.environment.specs.implementation,
            #     policy_network_hidden_size=config.policy_network.hidden_size,
            # )

            model_id = f"{run_session.run_id}_model"

            # Initializing a model
            model, _version_info = await self.create_and_publish_initial_version(
                model_id,
                environment_specs=config.environment.specs,
                policy_network_hidden_size=config.policy_network.hidden_size,
            )

            # Helper function to create a trial configuration
            def create_trial_config(trial_idx):
                env_params = copy.deepcopy(config.environment)
                env_params.config.seed = env_params.config.seed + trial_idx

                return TrialConfig(
                    run_id=run_session.run_id,
                    environment=env_params,
                    actors=actors_params,
                )


            # Rollout a bunch of trials
            async for (
                ############ TUTORIAL STEP 4 ############
                step_idx,
                step_timestamp,
                ##########################################
                _trial_id,
                _tick_id,
                sample,
            ) in run_session.start_trials_and_wait_for_termination(
                trial_configs=[create_trial_config(trial_idx) for trial_idx in range(config.training.trial_count)],
                max_parallel_trials=config.training.max_parallel_trials,
            ):
                ############ TUTORIAL STEP 4 ############
                (_demonstration, observation, action) = sample
                # Can be uncommented to only use samples coming from the teacher
                # (demonstration, observation, action) = sample
                # if not demonstration:
                #     continue
                observations.append(observation)
                actions.append(action)

                if len(observations) < config.training.batch_size:
                    continue


                # Publish the newly trained version every 100 steps
                # if step_idx % 100 == 0:
                #     version_info = await self.publish_version(model_id, model)
                #
                #     xp_tracker.log_metrics(
                #         step_timestamp,
                #         step_idx,
                #         model_version_number=version_info["version_number"],
                #         loss=loss,
                #         total_samples=len(observations),
                #     )
                ##########################################

        return {
            "simple_sb3_training": (
                # sample_producer_impl,
                run_impl,
                SB3TrainingRunConfig(
                    environment=EnvironmentParams(
                        specs=EnvironmentSpecs(implementation="gym/LunarLander-v2", num_input=8, num_action=4),
                        config=EnvironmentConfig(seed=12, framestack=1, render=True, render_width=256),
                    ),
                    ############ TUTORIAL STEP 4 ############
                    training=SimpleSB3TrainingRunConfig(
                        trial_count=100,
                        max_parallel_trials=1,
                        discount_factor=0.95,
                        learning_rate=0.01,
                    ),
                    ##########################################
                    policy_network=MLPNetworkConfig(hidden_size=64),
                ),
            )
        }
