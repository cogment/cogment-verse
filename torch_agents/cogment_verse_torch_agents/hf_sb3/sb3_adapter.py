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

import torch
from cogment.api.common_pb2 import TrialState
from cogment_verse import AgentAdapter

# from cogment_verse import MlflowExperimentTracker
from cogment_verse_torch_agents.utils.tensors import tensor_from_cog_obs
from data_pb2 import (
    ActorParams,
    AgentAction,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentSpecs,
    TrialConfig,
    SimpleSB3TrainingRunConfig,
)
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

# from stable_baselines3.common.evaluation import evaluate_policy

SimpleSB3Model = namedtuple("SimpleSB3Model", ["model_id", "version_number", "policy_network"])

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

    def _create_actor_implementations(self):
        async def impl(actor_session):
            actor_session.start()

            # config = actor_session.config

            checkpoint = load_from_hub(
                repo_id="ThomasSimonini/ppo-LunarLander-v2",
                filename="ppo-LunarLander-v2.zip",
            )
            model = PPO.load(checkpoint)

            @torch.no_grad()
            def compute_action(event):
                obs = tensor_from_cog_obs(event.observation.snapshot, dtype=self._dtype)
                obs = torch.unsqueeze(obs, dim=0)
                action = model.predict(obs)

                return action

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    action = await self.run_async(compute_action, event)
                    actor_session.do_action(AgentAction(discrete_action=action[0]))

        return {
            "simple_sb3": (impl, ["agent"]),
        }

    def _create_run_implementations(self):
        async def sample_producer_impl(run_sample_producer_session):
            assert run_sample_producer_session.count_actors() == 1

            async for sample in run_sample_producer_session.get_all_samples():
                if sample.get_trial_state() == TrialState.ENDED:
                    break

                log.info("Got raw sample")

        async def run_impl(run_session):
            config = run_session.config
            assert config.environment.specs.num_players == 1

            # Helper function to create a trial configuration
            def create_trial_config(trial_idx):
                env_params = copy.deepcopy(config.environment)
                env_params.config.seed = env_params.config.seed + trial_idx

                agent_actor_params = ActorParams(
                    name="agent_1",
                    actor_class="agent",
                    implementation="simple_sb3",
                    agent_config=AgentConfig(
                        run_id=run_session.run_id,
                        environment_specs=env_params.specs,
                    ),
                )

                return TrialConfig(
                    run_id=run_session.run_id,
                    environment=env_params,
                    actors=[agent_actor_params],
                )

            # Rollout a bunch of trials
            async for (
                # step_idx,
                # step_timestamp,
                _trial_id,
                _tick_id,
                sample,
            ) in run_session.start_trials_and_wait_for_termination(
                trial_configs=[create_trial_config(trial_idx) for trial_idx in range(config.training.trial_count)],
                max_parallel_trials=config.training.max_parallel_trials,
            ):
                log.info(f"Got sample {sample}")

        # checkpoint = load_from_hub(
        #     repo_id="ThomasSimonini/ppo-LunarLander-v2",
        #     filename="ppo-LunarLander-v2.zip",
        # )
        return {
            "simple_sb3_training": (
                sample_producer_impl,
                run_impl,
                SimpleSB3TrainingRunConfig(
                    environment=EnvironmentParams(
                        specs=EnvironmentSpecs(implementation="gym/LunarLander-v2", num_input=8, num_action=4),
                        config=EnvironmentConfig(seed=12, framestack=1, render=True, render_width=256),
                    ),
                    policy_network="ppo",
                ),
            )
        }
