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

from google.protobuf.json_format import MessageToDict

from data_pb2 import (
    ActorConfig,
    ActorParams,
    EnvironmentConfig,
    EnvironmentParams,
    TrialConfig,
)
from cogment_verse import MlflowExperimentTracker

import logging

# pylint: disable=protected-access
log = logging.getLogger(__name__)


def create_training_run(agent_adapter):
    async def training_run(run_session):
        run_id = run_session.run_id
        config = run_session.config

        run_xp_tracker = MlflowExperimentTracker(run_session.params_name, run_id)

        try:
            # Initialize Alice Agent
            alice_id = f"{run_id}_alice"
            alice_version_number = 1
            # TBD: hyperparameters, model kwargs
            # alice_kwargs = MessageToDict(config.model_kwargs, preserving_proto_field_name=True)
            alice, _ = await agent_adapter.create_and_publish_initial_version(
                alice_id,
                **{
                    "obs_dim": config.actor.config.num_input,
                    "act_dim": config.actor.config.num_action,
                #     "max_replay_buffer_size": config.max_replay_buffer_size,
                #     "lr": config.learning_rate,
                #     "gamma": config.discount_factor,
                },
                # **alice_kwargs,
            )


            # Initialize Bob Agent
            bob_id = f"{run_id}_bob"
            bob_version_number = 1
            # TBD: hyperparameters, model kwargs
            # bob_kwargs = MessageToDict(config.model_kwargs, preserving_proto_field_name=True)
            bob, _ = await agent_adapter.create_and_publish_initial_version(
                bob_id,
                **{
                    "obs_dim": config.actor.config.num_input,
                    "act_dim": config.actor.config.num_action,
                #     "max_replay_buffer_size": config.max_replay_buffer_size,
                #     "lr": config.learning_rate,
                #     "gamma": config.discount_factor,
                },
                # **bob_kwargs,
            )

            # run_xp_tracker.log_params(alice._params)
            # run_xp_tracker.log_params(bob._params)

            trials_completed = 0
            all_trials_reward = 0

            # create Alice_config
            alice_configs = [
                ActorParams(
                    name=f"selfplayRL_Alice",
                    actor_class="agent",
                    implementation=config.actor.implementation,
                    config=ActorConfig(
                        model_id=alice_id,
                        model_version=alice_version_number,
                        run_id=run_id,
                        environment_implementation=config.environment.implementation,
                        num_input=config.actor.config.num_input,
                        num_action=config.actor.config.num_action,
                    ),
                )
            ]

            # create Bob_config
            bob_configs = [
                ActorParams(
                    name=f"selfplayRL_Bob",
                    actor_class="agent",
                    implementation=config.actor.implementation,
                    config=ActorConfig(
                        model_id=bob_id,
                        model_version=bob_version_number,
                        run_id=run_id,
                        environment_implementation=config.environment.implementation,
                        num_input=config.actor.config.num_input,
                        num_action=config.actor.config.num_action,
                    ),
                )
            ]

            trial_configs = [
                TrialConfig(
                    run_id=run_id,
                    environment=EnvironmentParams(
                        implementation=config.environment.implementation,
                        config=EnvironmentConfig(
                            player_count=config.environment.config.player_count,
                            run_id=run_id,
                            render=False,
                            render_width=config.environment.config.render_width,
                            flatten=config.environment.config.flatten,
                            framestack=config.environment.config.framestack,
                        ),
                    ),
                    actors=alice_configs + bob_configs,
                )
                for _ in range(config.rollout.epoch_trial_count)
            ]

            bob_samples = []
            alice_samples = []
            for epoch in range(config.rollout.epoch_count):
                # Rollout Alice trials
                async for (
                    step_idx,
                    step_timestamp,
                    _trial_id,
                    _tick_id,
                    sample,
                ) in run_session.start_trials_and_wait_for_termination(
                    trial_configs=trial_configs,
                    max_parallel_trials=config.rollout.max_parallel_trials,
                ):
                    if sample.current_player == 0: # bob's sample
                        bob_samples.append(sample)
                        # penalize/reward alice if bob does/doesn't achieve goal
                        if sample.player_done:
                            if int(sample.reward) == 5:
                                alice_samples[-1] = alice_samples[-1]._replace(reward=-2.0)
                            else:
                                alice_samples[-1] = alice_samples[-1]._replace(reward=3.0)
                    else: # alice's sample
                        alice_samples.append(sample)

                # Train Bob
                # bob.learn()
                # Train Alice
                # alice.learn()x

                alice_version_info = await agent_adapter.publish_version(alice_id, alice)
                bob_version_info = await agent_adapter.publish_version(bob_id, bob)

            run_xp_tracker.terminate_success()

        except Exception:
            run_xp_tracker.terminate_failure()
            raise

    return training_run
