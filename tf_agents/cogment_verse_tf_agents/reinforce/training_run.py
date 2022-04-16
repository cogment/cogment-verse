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

from cogment_verse import MlflowExperimentTracker
from data_pb2 import AgentConfig, ActorParams, EnvironmentConfig, EnvironmentParams, TrialConfig
from google.protobuf.json_format import MessageToDict

# pylint: disable=protected-access
log = logging.getLogger(__name__)


def create_training_run(agent_adapter):
    async def training_run(run_session):
        run_id = run_session.run_id
        config = run_session.config

        run_xp_tracker = MlflowExperimentTracker(run_session.params_name, run_id)

        try:
            # Initializing a model
            model_id = f"{run_id}_model"
            model_version_number = 1
            model_kwargs = MessageToDict(config.model_kwargs, preserving_proto_field_name=True)
            model, _ = await agent_adapter.create_and_publish_initial_version(
                model_id,
                environment_specs=config.environment.specs,
                **{
                    "max_replay_buffer_size": config.max_replay_buffer_size,
                    "lr": config.learning_rate,
                    "gamma": config.discount_factor,
                },
                **model_kwargs,
            )

            run_xp_tracker.log_params(model._params)

            trials_completed = 0
            all_trials_reward = 0

            # Create config for the actor
            actor_configs = [
                ActorParams(
                    name=f"reinforce_player_{player_idx}",
                    actor_class="agent",
                    implementation=config.agent_implementation,
                    agent_config=AgentConfig(
                        run_id=run_id,
                        model_id=model_id,
                        model_version=model_version_number,
                        environment_specs=config.environment.specs,
                    ),
                )
                for player_idx in range(config.environment.specs.num_players)
            ]

            # Create configs for trials
            trial_configs = [
                TrialConfig(
                    run_id=run_id,
                    environment=EnvironmentParams(
                        specs=config.environment.specs,
                        config=EnvironmentConfig(
                            run_id=run_id,
                            render=False,
                            render_width=config.environment.config.render_width,
                            flatten=config.environment.config.flatten,
                            framestack=config.environment.config.framestack,
                        ),
                    ),
                    actors=actor_configs,
                )
                for _ in range(config.total_trial_count)
            ]

            async for (
                step_idx,
                step_timestamp,
                _trial_id,
                _tick_id,
                sample,
            ) in run_session.start_trials_and_wait_for_termination(
                trial_configs=trial_configs,
                max_parallel_trials=config.max_parallel_trials,
            ):
                model.consume_training_sample(sample.player_sample)

                # Check if last sample
                if sample.player_sample[-1]:

                    # Log trial reward stats
                    trials_completed += 1
                    all_trials_reward += sample.trial_cumulative_reward

                    run_xp_tracker.log_metrics(
                        step_timestamp,
                        step_idx,
                        trial_total_reward=sample.trial_cumulative_reward,
                        trials_completed=trials_completed,
                        mean_trial_reward=all_trials_reward / trials_completed,
                    )

                    # Train agent
                    hyperparams = model.learn()

                    # Log metrics about the published model
                    version_info = await agent_adapter.publish_version(model_id, model)
                    run_xp_tracker.log_metrics(
                        step_timestamp,
                        step_idx,
                        **hyperparams,
                        model_published_version=version_info["version_number"],
                        trials_completed=trials_completed,
                    )

                    log.info(
                        f"[{run_session.params_name}/{run_id}] {model_id}@v{version_info['version_number']} completed;"
                        f" {trials_completed} trials completed"
                    )

            run_xp_tracker.terminate_success()

        except Exception:
            run_xp_tracker.terminate_failure()
            raise

    return training_run
