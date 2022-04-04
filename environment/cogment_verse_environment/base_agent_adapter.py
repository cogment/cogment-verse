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

import logging
import copy

import numpy as np
import cogment

from cogment.api.common_pb2 import TrialState
from cogment_verse import AgentAdapter, MlflowExperimentTracker
from data_pb2 import (
    ActorParams,
    AgentAction,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentSpecs,
    HumanConfig,
    HumanRole,
    PlayRunConfig,
    TrialConfig,
)


log = logging.getLogger(__name__)


def extend_actor_config(actor_config_template, run_id, environment_specs):
    config = AgentConfig()
    config.CopyFrom(actor_config_template)
    config.run_id = run_id
    config.environment_specs.CopyFrom(environment_specs)
    return config


# pylint: disable=arguments-differ
class BaseAgentAdapter(AgentAdapter):
    def _create_actor_implementations(self):
        async def random_impl(actor_session):
            actor_session.start()

            config = actor_session.config

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    action = np.random.default_rng().integers(0, config.environment_specs.num_action)
                    actor_session.do_action(AgentAction(discrete_action=action))

        return {
            "random": (random_impl, ["agent"]),
        }

    def _create_run_implementations(self):
        async def total_rewards_producer_impl(run_sample_producer_session):
            actors_total_rewards = [0.0 for actor_idx in range(run_sample_producer_session.count_actors())]
            async for sample in run_sample_producer_session.get_all_samples():
                if sample.get_trial_state() == TrialState.ENDED:
                    break

                for actor_idx in range(run_sample_producer_session.count_actors()):
                    actors_total_rewards[actor_idx] += sample.get_actor_reward(actor_idx, 0.0)

            run_sample_producer_session.produce_training_sample(actors_total_rewards)

        async def play_impl(run_session):
            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

            config = run_session.config
            # We ignore additional actor configs
            if config.environment.specs.num_players > len(config.actors):
                raise RuntimeError(
                    f"Expecting at least {config.environment.specs.num_players} configured actors, got {len(config.actors)}"
                )

            actors_params = [
                ActorParams(
                    name=actor_params.name,
                    actor_class=actor_params.actor_class,
                    implementation=actor_params.implementation,
                    agent_config=extend_actor_config(
                        actor_config_template=actor_params.agent_config,
                        run_id=run_session.run_id,
                        environment_specs=config.environment.specs,
                    ),
                )
                for actor_params in config.actors[: config.environment.specs.num_players]
            ]

            xp_tracker.log_params(
                config.environment.config,
                **{
                    f"actor_{actor_idx}_implementation": actor_params.implementation
                    for actor_idx, actor_params in enumerate(actors_params)
                },
                **{
                    f"actor_{actor_idx}_model_id": actor_params.agent_config.model_id
                    for actor_idx, actor_params in enumerate(actors_params)
                },
                **{
                    f"actor_{actor_idx}_model_version": actor_params.agent_config.model_version
                    for actor_idx, actor_params in enumerate(actors_params)
                },
                environment=config.environment.specs.implementation,
            )

            if config.observer:
                # Add an observer agent
                actors_params.append(
                    ActorParams(
                        name="web_actor",
                        actor_class="teacher_agent",
                        implementation="client",
                        human_config=HumanConfig(
                            run_id=run_session.run_id,
                            environment_specs=config.environment.specs,
                            role=HumanRole.OBSERVER,
                        ),
                    )
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
                step_idx,
                step_timestamp,
                _trial_id,
                _tick_id,
                sample,
            ) in run_session.start_trials_and_wait_for_termination(
                trial_configs=[create_trial_config(trial_idx) for trial_idx in range(config.trial_count)],
                max_parallel_trials=1,
            ):
                xp_tracker.log_metrics(
                    step_timestamp,
                    step_idx,
                    **{
                        f"actor_{actor_idx}_reward": sample[actor_idx]
                        for actor_idx in range(config.environment.specs.num_players)
                    },
                    total_reward=sum(sample),
                )

        return {
            "play": (
                total_rewards_producer_impl,
                play_impl,
                PlayRunConfig(
                    environment=EnvironmentParams(
                        specs=EnvironmentSpecs(
                            implementation="gym/LunarLander-v2", num_input=8, num_action=4, num_players=1
                        ),
                        config=EnvironmentConfig(seed=12, framestack=1, render=True, render_width=256),
                    ),
                    actors=[],
                    trial_count=5,
                ),
            )
        }
