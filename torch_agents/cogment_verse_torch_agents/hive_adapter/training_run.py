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

import time

import torch
import numpy as np
from prometheus_client import Summary, Gauge
from google.protobuf.json_format import MessageToDict

from data_pb2 import (
    ActorConfig,
    ActorParams,
    EnvironmentConfig,
    EnvironmentParams,
    TrialConfig,
    NDArray,
)
from cogment_verse import MlflowExperimentTracker
from cogment_verse.utils import sizeof_fmt, throttle
from cogment_verse_torch_agents.third_party.hive.utils.schedule import (
    PeriodicSchedule,
    LinearSchedule,
    SwitchSchedule,
    CosineSchedule,
)

import logging

# pylint: disable=protected-access

TRAINING_ADD_SAMPLE_TIME = Summary("training_add_sample_seconds", "Time spent adding samples in the replay buffer")
TRAINING_SAMPLE_BATCH_TIME = Summary(
    "training_sample_batch_seconds",
    "Time spent sampling the replay buffer to create a batch",
)
TRAINING_LEARN_TIME = Summary("training_learn_seconds", "Time spent learning")
TRAINING_REPLAY_BUFFER_SIZE = Gauge("replay_buffer_size", "Size of the replay buffer")

log = logging.getLogger(__name__)


def create_progress_logger(params_name, run_id, total_trial_count):
    @throttle(seconds=20)
    def handle_progress(_launched_trial_count, finished_trial_count):
        log.info(
            f"[{params_name}/{run_id}] {finished_trial_count} ({finished_trial_count/total_trial_count:.1%}) trials finished"
        )

    return handle_progress

def serialize_np_array(np_array):
    return NDArray(shape=np_array.shape, dtype=str(np_array.dtype), data=np_array.tobytes())

def create_training_run(agent_adapter):
    async def training_run(run_session):
        run_id = run_session.run_id

        config2 = run_session.config

        run_xp_tracker = MlflowExperimentTracker(run_session.params_name, run_id)

        try:
            # Initializing a model
            model_id = f"{run_id}_model"

            model_kwargs = MessageToDict(config2.model_kwargs, preserving_proto_field_name=True)
            model_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

            model, _ = await agent_adapter.create_and_publish_initial_version(
                model_id,
                impl_name=config2.agent_implementation,
                **{
                    "obs_dim": config2.num_input,
                    "act_dim": config2.num_action,
                    "epsilon_schedule": LinearSchedule(1, config2.epsilon_min, config2.epsilon_steps),
                    "learn_schedule": SwitchSchedule(False, True, 1),
                    "target_net_update_schedule": PeriodicSchedule(False, True, config2.target_net_update_schedule),
                    "lr_schedule": CosineSchedule(0.0, config2.learning_rate, config2.lr_warmup_steps),
                    "max_replay_buffer_size": config2.max_replay_buffer_size,
                },
                **model_kwargs,
            )
            run_xp_tracker.log_params(
                model._params,
                player_count=config2.player_count,
                batch_size=config2.batch_size,
                model_publication_interval=config2.model_publication_interval,
                model_archive_interval_multiplier=config2.model_archive_interval_multiplier,
                environment=config2.environment_implementation,
                agent_implmentation=config2.agent_implementation,
            )

            model_publication_schedule = PeriodicSchedule(False, True, config2.model_publication_interval)
            model_archive_schedule = PeriodicSchedule(
                False,
                True,
                config2.model_archive_interval_multiplier * config2.model_publication_interval,
            )

            training_step = 0
            samples_seen = 0
            samples_generated = 0
            trials_completed = 0
            all_trials_reward = 0
            start_time = time.time()

            # Create the config for the player agents
            player_actor_configs = [
                ActorParams(
                    name=f"agent_player_{player_idx}",
                    actor_class="agent",
                    implementation=config2.agent_implementation,
                    config=ActorConfig(
                        model_id=model_id,
                        model_version=np.random.randint(-100, -1),  # TODO this actually won't work anymore
                        run_id=run_id,
                        environment_implementation=config2.environment_implementation,
                        num_input=config2.num_input,
                        num_action=config2.num_action,
                    ),
                )
                for player_idx in range(config2.player_count)
            ]
            # for self-play, randomly select one player to use latest model version
            # if there is only one player then it will always use the latest
            distinguished_actor = np.random.randint(0, config2.player_count)
            player_actor_configs[distinguished_actor].config.model_version = -1

            self_play_trial_configs = [
                TrialConfig(
                    run_id=run_id,
                    environment=EnvironmentParams(
                        implementation=config2.environment_implementation,
                        config=EnvironmentConfig(
                            player_count=config2.player_count,
                            run_id=run_id,
                            render=False,
                            render_width=config2.render_width,
                            flatten=config2.flatten,
                            framestack=config2.framestack,
                        ),
                    ),
                    actors=player_actor_configs,
                    distinguished_actor=distinguished_actor,
                )
                for _ in range(config2.total_trial_count - config2.demonstration_count)
            ]

            demonstration_trial_configs = []
            if config2.demonstration_count > 0:
                # create the config for the teacher agent
                teacher_actor_config = ActorParams(
                    name="web_actor",
                    actor_class="teacher_agent",
                    implementation="client",
                    config=ActorConfig(
                        run_id=run_id,
                        environment_implementation=config2.environment_implementation,
                        num_input=config2.num_input,
                        num_action=config2.num_action,
                    ),
                )
                demonstration_trial_configs = [
                    TrialConfig(
                        run_id=run_id,
                        environment=EnvironmentParams(
                            implementation=config2.environment_implementation,
                            config=EnvironmentConfig(
                                player_count=config2.player_count,
                                run_id=run_id,
                                render=True,
                                render_width=config2.render_width,
                                flatten=config2.flatten,
                                framestack=config2.framestack,
                            ),
                        ),
                        actors=[*player_actor_configs, teacher_actor_config],
                        distinguished_actor=distinguished_actor,
                    )
                    for _ in range(config2.demonstration_count)
                ]

            def train_model():
                training_batch = None
                with TRAINING_SAMPLE_BATCH_TIME.time():
                    training_batch = model.sample_training_batch(config2.batch_size)

                with TRAINING_LEARN_TIME.time():
                    info = model.learn(training_batch, update_schedule=True)
                return info, training_batch

            def get_samples_seen(training_batch):
                for key in training_batch.keys():
                    num_samples_seen = training_batch[key].shape[0]
                    break
                return num_samples_seen

            async def archive_model(
                model_archive_schedule,
                model_publication_schedule,
                step_timestamp,
                step_idx,
                training_batch,
                info,
            ):

                archive = model_archive_schedule.update()
                publish = model_publication_schedule.update()
                if archive or publish:
                    version_info = await agent_adapter.publish_version(model_id, model, archived=archive)
                    version_number = version_info["version_number"]
                    version_data_size = int(version_info["data_size"])

                    # Log metrics about the published model
                    run_xp_tracker.log_metrics(
                        step_timestamp,
                        step_idx,
                        info,
                        epsilon=model._epsilon_schedule.get_value(),
                        replay_buffer_size=model.replay_buffer_size(),
                        batch_reward=training_batch["rewards"].mean(),
                        batch_done=training_batch["done"].mean(),
                        model_published_version=version_number,
                        training_step=training_step,
                        training_samples_seen=samples_seen,
                        samples_generated=samples_generated,
                        episodes_per_sec=trials_completed / (time.time() - start_time),
                    )
                    verb = "archived" if archive else "published"
                    log.info(
                        f"[{run_session.params_name}/{run_id}] {model_id}@v{version_number} {verb} after {run_session.count_steps()} steps ({sizeof_fmt(version_data_size)})"
                    )

            async def run_trials(trial_configs, max_parallel_trials):
                nonlocal samples_generated
                nonlocal training_step
                nonlocal samples_seen
                nonlocal trials_completed
                nonlocal all_trials_reward
                nonlocal start_time

                async for (
                    step_idx,
                    step_timestamp,
                    _trial_id,
                    _tick_id,
                    sample,
                ) in run_session.start_trials_and_wait_for_termination(
                    trial_configs=trial_configs,
                    max_parallel_trials=max_parallel_trials,
                    on_progress=create_progress_logger(run_session.params_name, run_id, config2.total_trial_count),
                ):
                    if sample.trial_total_reward is not None:
                        # This is a sample from a end of a trial

                        trials_completed += 1
                        all_trials_reward += sample.trial_total_reward

                        run_xp_tracker.log_metrics(
                            step_timestamp,
                            step_idx,
                            trial_total_reward=sample.trial_total_reward,
                            trials_completed=trials_completed,
                            mean_trial_reward=all_trials_reward / trials_completed,
                        )

                    samples_generated += 1

                    with TRAINING_ADD_SAMPLE_TIME.time():
                        model.consume_training_sample(sample.current_player_sample)

                    TRAINING_REPLAY_BUFFER_SIZE.set(model.replay_buffer_size())

                    if sample.current_player_sample[-1] and model.replay_buffer_size() > config2.batch_size:
                        info, training_batch = train_model()
                        samples_seen += get_samples_seen(training_batch)
                        training_step += 1
                        model.reset_replay_buffer()

                        await archive_model(
                            model_archive_schedule,
                            model_publication_schedule,
                            step_timestamp,
                            step_idx,
                            training_batch,
                            info,
                        )

                    elif (
                        model.replay_buffer_size() > config2.min_replay_buffer_size
                        and model.replay_buffer_size() > config2.batch_size
                    ):
                        info, training_batch = train_model()
                        samples_seen += get_samples_seen(training_batch)
                        training_step += 1

                        await archive_model(
                            model_archive_schedule,
                            model_publication_schedule,
                            step_timestamp,
                            step_idx,
                            training_batch,
                            info,
                        )

                log.info(
                    f"[{run_session.params_name}/{run_id}] done, {model.replay_buffer_size()} samples gathered over {run_session.count_steps()} steps"
                )

            if demonstration_trial_configs:
                await run_trials(
                    demonstration_trial_configs,
                    max_parallel_trials=config2.max_parallel_trials,
                )
            if self_play_trial_configs:
                await run_trials(self_play_trial_configs, max_parallel_trials=config2.max_parallel_trials)

            run_xp_tracker.terminate_success()

        except Exception:
            run_xp_tracker.terminate_failure()
            raise

    return training_run