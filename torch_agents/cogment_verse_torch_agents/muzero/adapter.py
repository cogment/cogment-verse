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

from collections import defaultdict

from data_pb2 import (
    MuZeroTrainingRunConfig,
    MuZeroTrainingConfig,
    AgentAction,
    TrialConfig,
    TrialActor,
    EnvConfig,
    ActorConfig,
    MLPNetworkConfig,
    NDArray,
)

from cogment_verse import AgentAdapter
from cogment_verse import MlflowExperimentTracker
from cogment_verse_torch_agents.wrapper import proto_array_from_np_array
from cogment_verse_torch_agents.muzero.replay_buffer import Episode, TrialReplayBuffer, EpisodeBatch

from cogment.api.common_pb2 import TrialState
import cogment

import logging
import torch
import numpy as np


log = logging.getLogger(__name__)

from .agent import MuZeroAgent

# pylint: disable=arguments-differ


class RunningStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self._running_stats = defaultdict(int)
        self._running_counts = defaultdict(int)

    def update(self, info):
        for key, val in info.items():
            self._running_stats[key] += val
            self._running_counts[key] += 1

    def get(self):
        return {key: self._running_stats[key] / count for key, count in self._running_counts.items()}


DEFAULT_MUZERO_TRAINING_CONFIG = MuZeroTrainingConfig(
    discount_rate=0.99,
    learning_rate=1e-4,
    weight_decay=1e-3,
    bootstrap_steps=20,
    representation_dim=32,
    hidden_dim=128,
    hidden_layers=2,
    projector_hidden_dim=128,
    projector_hidden_layers=1,
    projector_dim=64,
    mcts_depth=3,
    mcts_samples=8,
    ucb_c1=1.25,
    ucb_c2=10000.0,
    batch_size=16,
    exploration_alpha=0.5,
    exploration_epsilon=0.25,
    rollout_length=2,
    rmin=-100.0,
    rmax=100.0,
    vmin=-300.0,
    vmax=300.0,
    rbins=16,
    vbins=16,
    epoch_count=10,
    epoch_trial_count=100,
    max_parallel_trials=4,
    mcts_temperature=0.99,
    max_replay_buffer_size=20000,
    min_replay_buffer_size=200,
    log_frequency=20,
)


class MuZeroAgentAdapter(AgentAdapter):
    def tensor_from_cog_obs(self, cog_obs, device=None):
        pb_array = cog_obs.vectorized
        np_array = np.frombuffer(pb_array.data, dtype=pb_array.dtype).reshape(*pb_array.shape)
        return torch.tensor(np_array, dtype=self._dtype, device=device)

    @staticmethod
    def decode_cog_action(cog_action):
        action = cog_action.discrete_action
        policy = np.frombuffer(cog_action.policy.data, dtype=cog_action.policy.dtype).reshape(*cog_action.policy.shape)
        value = cog_action.value
        return action, policy, value

    def __init__(self):
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float

    def _create(self, model_id, *, obs_dim, act_dim, device, training_config):
        return MuZeroAgent(obs_dim=obs_dim, act_dim=act_dim, device=device, training_config=training_config)

    def _load(self, model_id, version_number, version_user_data, model_data_f):
        return MuZeroAgent.load(model_data_f, self._device)

    def _save(self, model, model_data_f):
        assert isinstance(model, MuZeroAgent)
        model.save(model_data_f)
        return {}

    def _create_actor_implementations(self):
        async def _single_agent_muzero_actor_implementation(actor_session):
            actor_session.start()

            config = actor_session.config

            agent, _ = await self.retrieve_version(config.model_id, config.model_version)

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    obs = self.tensor_from_cog_obs(event.observation.snapshot)
                    action, policy, value = agent.act(obs)
                    actor_session.do_action(
                        AgentAction(discrete_action=action, policy=proto_array_from_np_array(policy), value=value)
                    )

        return {
            "muzero_mlp": (_single_agent_muzero_actor_implementation, ["agent"]),
        }

    def _create_run_implementations(self):
        """
        Create all the available run implementation for this adapter
        Returns:
            dict[impl_name: string, (sample_producer_impl: Callable, run_impl: Callable, default_run_config)]: key/value definition for the available run implementations.
        """

        async def _single_agent_muzero_sample_producer_implementation(run_sample_producer_session):
            return await single_agent_muzero_sample_producer_implementation(self, run_sample_producer_session)

        async def _single_agent_muzero_run_implementation(run_session):
            return await single_agent_muzero_run_implementation(self, run_session)

        return {
            "muzero_mlp_training": (
                _single_agent_muzero_sample_producer_implementation,
                _single_agent_muzero_run_implementation,
                MuZeroTrainingRunConfig(
                    environment=EnvConfig(
                        seed=12, env_type="gym", env_name="CartPole-v0", player_count=1, framestack=1
                    ),
                    training=DEFAULT_MUZERO_TRAINING_CONFIG,
                ),
            )
        }


async def single_agent_muzero_actor_implementation(agent_adapter, actor_session):
    actor_session.start()

    config = actor_session.config

    agent, _ = await agent_adapter.retrieve_version(config.model_id, config.model_version)

    async for event in actor_session.event_loop():
        if event.observation and event.type == cogment.EventType.ACTIVE:
            obs = agent_adapter.tensor_from_cog_obs(event.observation.snapshot)
            action, policy, value = agent.act(obs)
            actor_session.do_action(
                AgentAction(discrete_action=action, policy=proto_array_from_np_array(policy), value=value)
            )


async def single_agent_muzero_sample_producer_implementation(agent_adapter, run_sample_producer_session):
    assert run_sample_producer_session.count_actors() == 1
    state = None
    step = 0
    total_reward = 0

    async for sample in run_sample_producer_session.get_all_samples():
        next_state = agent_adapter.tensor_from_cog_obs(sample.get_actor_observation(0))
        done = sample.get_trial_state() == TrialState.ENDED

        if state is not None:
            total_reward += reward
            run_sample_producer_session.produce_training_sample(
                (
                    EpisodeBatch(
                        episode=0,
                        step=step,
                        state=state,
                        action=action,
                        rewards=reward,
                        next_state=next_state,
                        done=done,
                        target_policy=policy,
                        target_value=value,
                        priority=0.001,
                    ),
                    total_reward,
                )
            )

        if done:
            break

        step += 1
        state = next_state
        action, policy, value = agent_adapter.decode_cog_action(sample.get_actor_action(0))
        reward = sample.get_actor_reward(0)


async def single_agent_muzero_run_implementation(agent_adapter, run_session):
    xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

    # Initializing a model
    model_id = f"{run_session.run_id}_model"

    config = run_session.config
    assert config.environment.player_count == 1

    agent, version_info = await agent_adapter.create_and_publish_initial_version(
        model_id=model_id,
        obs_dim=config.actor.num_input,
        act_dim=config.actor.num_action,
        device=agent_adapter._device,
        training_config=config.training,
    )
    model_version_number = 1
    trials_completed = 0
    running_stats = RunningStats()

    xp_tracker.log_params(
        config.environment,
        config.actor,
        config.training,
    )

    total_samples = 0
    for epoch in range(config.training.epoch_count):
        epoch_last_step_idx = None
        epoch_last_step_timestamp = None
        async for (
            step_idx,
            step_timestamp,
            _trial_id,
            _tick_id,
            (sample, total_reward),
        ) in run_session.start_trials_and_wait_for_termination(
            trial_configs=[
                TrialConfig(
                    run_id=run_session.run_id,
                    environment_config=config.environment,
                    actors=[
                        TrialActor(
                            name="agent_1",
                            actor_class="agent",
                            implementation="muzero_mlp",
                            config=ActorConfig(
                                model_id=model_id,
                                model_version=model_version_number,
                                num_input=config.actor.num_input,
                                num_action=config.actor.num_action,
                                env_type=config.environment.env_type,
                                env_name=config.environment.env_name,
                            ),
                        )
                    ],
                )
                for trial_ids in range(config.training.epoch_trial_count)
            ],
            max_parallel_trials=config.training.max_parallel_trials,
        ):
            total_samples += 1
            agent.consume_training_sample(
                state=sample.state,
                action=sample.action,
                reward=sample.rewards,
                next_state=sample.next_state,
                done=sample.done,
                policy=sample.target_policy,
                value=sample.target_value,
            )
            total_samples += 1
            epoch_last_step_idx = step_idx
            epoch_last_step_timestamp = step_timestamp

            if sample.done:
                trials_completed += 1
                xp_tracker.log_metrics(
                    step_timestamp, step_idx, trial_total_reward=total_reward, trials_completed=trials_completed
                )

            if agent._replay_buffer.size() > config.training.min_replay_buffer_size:
                batch = agent.sample_training_batch(config.training.batch_size)
                priority, info = agent.learn(batch)
                running_stats.update(info)

                if total_samples % config.training.log_frequency == 0:
                    xp_tracker.log_metrics(
                        step_timestamp,
                        step_idx,
                        total_samples=total_samples,
                        **running_stats.get(),
                    )
                    running_stats.reset()

            # xp_tracker.log_metrics(step_timestamp, step_idx, total_reward=sum([r.item() for r in trial_reward]))

        # total_samples += len(observation)

        # Publish the newly trained version
        version_info = await agent_adapter.publish_version(model_id, agent)
        model_version_number = version_info["version_number"]
        xp_tracker.log_metrics(
            epoch_last_step_timestamp,
            epoch_last_step_idx,
            model_version_number=model_version_number,
            epoch=epoch,
        )
        log.info(
            f"[{run_session.params_name}/{run_session.run_id}] epoch #{epoch} finished ({total_samples} samples seen)"
        )
