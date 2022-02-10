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
from collections import namedtuple
import ctypes
import copy
import time

import logging
import torch
import torch.multiprocessing as mp
import numpy as np
import queue

from data_pb2 import (
    MuZeroTrainingRunConfig,
    MuZeroTrainingConfig,
    AgentAction,
    TrialConfig,
    ActorParams,
    EnvironmentConfig,
    EnvironmentParams,
    EnvironmentSpecs,
    AgentConfig,
)

from cogment_verse.utils import LRU
from cogment_verse import AgentAdapter
from cogment_verse import MlflowExperimentTracker
from cogment_verse_torch_agents.wrapper import np_array_from_proto_array, proto_array_from_np_array
from cogment_verse_torch_agents.muzero.replay_buffer import Episode, TrialReplayBuffer, EpisodeBatch
from cogment_verse_torch_agents.muzero.agent import MuZeroAgent
from cogment_verse_torch_agents.muzero.schedule import LinearScheduleWithWarmup
from cogment_verse_torch_agents.muzero.stats import RunningStats

from cogment.api.common_pb2 import TrialState
import cogment

from cogment_verse_torch_agents.muzero.reanalyze_worker import ReanalyzeWorker
from cogment_verse_torch_agents.muzero.replay_worker import ReplayBufferWorker
from cogment_verse_torch_agents.muzero.trial_worker import AgentTrialWorker
from cogment_verse_torch_agents.muzero.train_worker import TrainWorker


# pylint: disable=arguments-differ

log = logging.getLogger(__name__)


MuZeroSample = namedtuple("MuZeroSample", ["state", "action", "reward", "next_state", "done", "policy", "value"])

DEFAULT_MUZERO_TRAINING_CONFIG = MuZeroTrainingConfig(
    model_publication_interval=500,
    trial_count=1000,
    discount_rate=0.99,
    learning_rate=1e-4,
    weight_decay=1e-3,
    bootstrap_steps=20,
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
    rmin=0.0,  # proto default value issue
    rmax=0.0,  # proto default value issue
    vmin=0.0,  # proto default value issue
    vmax=0.0,  # proto default value issue
    rbins=16,
    vbins=16,
    max_parallel_trials=4,
    mcts_temperature=0.99,
    max_replay_buffer_size=20000,
    min_replay_buffer_size=200,
    log_interval=200,
    min_learning_rate=1e-6,
    lr_warmup_steps=1000,
    lr_decay_steps=1000000,
    epsilon_min=0.01,
    epsilon_decay_steps=100000,
    min_temperature=0.25,
    temperature_decay_steps=100000,
    s_weight=1e-2,
    v_weight=0.1,
    train_device="cpu",
    actor_device="cpu",
    reanalyze_device="cpu",
    reanalyze_workers=0,
)


class MuZeroAgentAdapter(AgentAdapter):
    def tensor_from_cog_obs(self, cog_obs, device=None):
        pb_array = cog_obs.vectorized
        np_array = np.frombuffer(pb_array.data, dtype=pb_array.dtype).reshape(*pb_array.shape)
        return torch.tensor(np_array, dtype=self._dtype, device=device)

    @staticmethod
    def decode_cog_policy_value(cog_action):
        policy = np.frombuffer(cog_action.policy.data, dtype=cog_action.policy.dtype).reshape(*cog_action.policy.shape)
        value = cog_action.value
        return policy, value

    @staticmethod
    def decode_cog_action(cog_action):
        action = cog_action.discrete_action
        return action

    def __init__(self):
        super().__init__()
        self._model_cache = LRU(2)  # memory issue?
        self._dtype = torch.float

    def _create(
        self,
        model_id,
        environment_specs,
        device,
        training_config,
        **kwargs,
    ):
        model = MuZeroAgent(
            obs_dim=environment_specs.num_input,
            act_dim=environment_specs.num_action,
            device=device,
            training_config=training_config,
        )
        model_user_data = {
            "environment_implementation": environment_specs.implementation,
            "num_input": environment_specs.num_input,
            "num_action": environment_specs.num_action,
            "training_config": training_config,
        }
        return model, model_user_data

    def _load(
        self,
        model_id,
        version_number,
        model_user_data,
        version_user_data,
        model_data_f,
        environment_specs,
        **kwargs,
    ):
        return MuZeroAgent.load(model_data_f, "cpu")

    def _save(self, model, model_user_data, model_data_f, environment_specs, epoch_idx=-1, total_samples=0, **kwargs):
        assert isinstance(model, MuZeroAgent)
        model.save(model_data_f)
        return {"epoch_idx": epoch_idx, "total_samples": total_samples}

    def _create_actor_implementations(self):
        async def _single_agent_muzero_actor_implementation(actor_session):
            actor_session.start()
            agent, _, _ = await self.retrieve_version(actor_session.config.model_id, -1)
            # agent = copy.deepcopy(agent)
            agent.set_device(actor_session.config.device)

            worker = AgentTrialWorker(agent, actor_session.config)
            worker.start()

            try:
                async for event in actor_session.event_loop():
                    assert worker.is_alive()
                    if event.observation and event.type == cogment.EventType.ACTIVE:
                        await worker.put_event(event)
                        action = await worker.get_action()
                        actor_session.do_action(action)
            finally:
                worker.terminate()

        return {
            "muzero_mlp": (_single_agent_muzero_actor_implementation, ["agent"]),
        }

    async def single_agent_muzero_sample_producer_implementation(self, run_sample_producer_session):
        # allow up to two players for human/expert intervention
        assert run_sample_producer_session.count_actors() in (1, 2)
        state = None
        step = 0
        total_reward = 0
        state, action, reward, policy, value = None, None, None, None, None
        player_override = None

        async for sample in run_sample_producer_session.get_all_samples():
            observation = sample.get_actor_observation(0)
            next_state = self.tensor_from_cog_obs(observation)
            done = sample.get_trial_state() == TrialState.ENDED
            player_override = (
                observation.current_player if observation.player_override == -1 else observation.player_override
            )

            if state is not None:
                total_reward += reward

                run_sample_producer_session.produce_training_sample(
                    (MuZeroSample(state, action, reward, next_state, done, policy, value), total_reward)
                )

            if done:
                break

            step += 1
            state = next_state
            action = self.decode_cog_action(sample.get_actor_action(player_override))
            policy, value = self.decode_cog_policy_value(sample.get_actor_action(observation.current_player))

            if action < 0:
                # todo(jonathan): investigate this, it shouldn't happen
                print("WARNING: override action is invalid, ignoring!")
                action = self.decode_cog_action(sample.get_actor_action(observation.current_player))
                reward = sample.get_actor_reward(observation.current_player)
            else:
                # BC for teacher intervention
                if player_override != observation.current_player:
                    policy = np.zeros_like(policy)
                    policy[:, action] = 1.0
                reward = sample.get_actor_reward(player_override)

            assert action >= 0

    async def single_agent_muzero_run_implementation(self, run_session):
        xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

        # Initializing a model
        model_id = f"{run_session.run_id}_model"

        config = run_session.config
        assert config.environment.specs.num_players == 1

        agent, version_info = await self.create_and_publish_initial_version(
            model_id=model_id,
            environment_specs=config.environment.specs,
            device="cpu",
            training_config=config.training,
        )

        num_reanalyze_workers = config.training.reanalyze_workers

        with mp.Manager() as manager:
            max_prefetch_batch = 128
            sample_queue = manager.Queue()
            priority_update_queue = manager.Queue()
            reanalyze_update_queue = manager.Queue()
            reanalyze_queue = manager.Queue(num_reanalyze_workers + 1)

            batch_queue = manager.Queue(max_prefetch_batch)
            # limit to small size so that training and sample generation don't get out of sync
            results_queue = manager.Queue(max_prefetch_batch)

            reward_distribution = copy.deepcopy(agent.muzero.reward_distribution).cpu()
            value_distribution = copy.deepcopy(agent.muzero.value_distribution).cpu()

            train_worker = TrainWorker(agent, batch_queue, results_queue, config)
            replay_buffer = ReplayBufferWorker(
                sample_queue,
                priority_update_queue,
                batch_queue,
                reanalyze_queue,
                reanalyze_update_queue,
                config.training,
                reward_distribution,
                value_distribution,
            )
            reanalyze_workers = [
                ReanalyzeWorker(
                    manager.Queue(2),
                    reanalyze_queue,
                    reanalyze_update_queue,
                    model_id,
                    config.training.reanalyze_device,
                    reward_distribution,
                    value_distribution,
                    config.training.threads_per_worker,
                )
                for i in range(num_reanalyze_workers)
            ]

            trials_completed = 0
            running_stats = RunningStats()

            xp_tracker.log_params(
                config.environment,
                config.actor,
                config.training,
            )

            trial_configs = make_trial_configs(run_session, config, model_id, -1)

            total_samples = 0
            training_step = 0
            run_total_reward = 0
            epoch_idx = 0

            workers = [train_worker, replay_buffer] + reanalyze_workers

            for worker in reanalyze_workers:
                worker.update_agent(agent)

            try:
                for worker in workers:
                    worker.start()

                start_time = time.time()

                sample_generator = run_session.start_trials_and_wait_for_termination(
                    trial_configs,
                    max_parallel_trials=config.training.max_parallel_trials,
                )

                async for _step, timestamp, trial_id, _tick, (sample, total_reward) in sample_generator:
                    replay_buffer.add_sample(trial_id, sample)
                    total_samples += 1

                    for worker in workers:
                        assert worker.is_alive()

                    if sample.done:
                        trials_completed += 1
                        xp_tracker.log_metrics(
                            timestamp, total_samples, trial_total_reward=total_reward, trials_completed=trials_completed
                        )
                        run_total_reward += total_reward

                    if replay_buffer.size() <= config.training.min_replay_buffer_size:
                        continue

                    while not results_queue.empty():
                        try:
                            info, new_agent = results_queue.get_nowait()
                            if new_agent is not None:
                                epoch_idx += 1
                                agent = new_agent
                                version_info = await self.publish_version(
                                    model_id,
                                    agent,
                                    epoch_idx=epoch_idx,
                                    total_samples=total_samples,
                                    environment_specs=config.environment.specs,
                                )
                                for worker in reanalyze_workers:
                                    worker.update_agent(agent)
                        except queue.Empty:
                            continue

                        training_step += 1
                        info["model_version"] = version_info["version_number"]
                        info["training_step"] = training_step
                        info["mean_trial_reward"] = run_total_reward / max(1, trials_completed)
                        info["samples_per_sec"] = total_samples / max(1, time.time() - start_time)
                        info["reanalyzed_samples"] = sum([worker.reanalyzed_samples() for worker in reanalyze_workers])
                        running_stats.update(info)

                    if total_samples % config.training.log_interval == 0:
                        xp_tracker.log_metrics(
                            timestamp,
                            total_samples,
                            total_samples=total_samples,
                            **running_stats.get(),
                        )
                        running_stats.reset()

            finally:
                for worker in workers:
                    worker.terminate()

            log.info(f"[{run_session.params_name}/{run_session.run_id}] finished ({total_samples} samples seen)")

    def _create_run_implementations(self):
        """
        Create all the available run implementation for this adapter
        Returns:
            dict[impl_name: string, (sample_producer_impl: Callable, run_impl: Callable, default_run_config)]: key/value definition for the available run implementations.
        """
        return {
            "muzero_mlp_training": (
                self.single_agent_muzero_sample_producer_implementation,
                self.single_agent_muzero_run_implementation,
                MuZeroTrainingRunConfig(
                    environment=EnvironmentParams(
                        config=EnvironmentConfig(
                            seed=12,
                            framestack=1,
                            render=False,
                        ),
                        specs=EnvironmentSpecs(
                            implementation="gym/CartPole-v0", num_players=1, num_input=4, num_action=2
                        ),
                    ),
                    training=DEFAULT_MUZERO_TRAINING_CONFIG,
                ),
            )
        }


def make_trial_configs(run_session, config, model_id, model_version_number):
    def clone_config(config, render, seed):
        config = copy.deepcopy(config)
        config.render = render
        config.seed = seed
        return config

    actor_config = AgentConfig(
        run_id=run_session.run_id,
        model_id=model_id,
        model_version=model_version_number,
        device=config.training.actor_device,
        environment_specs=config.environment.specs,
        actor_index=0,
    )
    muzero_config = ActorParams(
        name="agent_1",
        actor_class="agent",
        implementation="muzero_mlp",
        agent_config=actor_config,
    )
    teacher_config = ActorParams(
        name="web_actor",
        actor_class="teacher_agent",
        implementation="client",
        agent_config=actor_config,  # todo: this needs to be modified to HumanConfig
    )
    demonstration_configs = [
        TrialConfig(
            run_id=run_session.run_id,
            environment=EnvironmentParams(
                config=clone_config(
                    config.environment.config,
                    seed=config.environment.config.seed + i + config.training.demonstration_trials,
                    render=True,
                ),
                specs=config.environment.specs,
            ),
            actors=[muzero_config, teacher_config],
        )
        for i in range(config.training.demonstration_trials)
    ]

    trial_configs = [
        TrialConfig(
            run_id=run_session.run_id,
            environment=EnvironmentParams(
                config=clone_config(
                    config.environment.config,
                    seed=config.environment.config.seed + i + config.training.demonstration_trials,
                    render=False,
                ),
                specs=config.environment.specs,
            ),
            actors=[muzero_config],
        )
        for i in range(config.training.trial_count - config.training.demonstration_trials)
    ]

    return demonstration_configs + trial_configs


@torch.no_grad()
def compute_targets(reward, value, reward_distribution, value_distribution):
    reward_probs = reward_distribution.compute_target(torch.tensor(reward)).cpu()
    value_probs = value_distribution.compute_target(torch.tensor(value)).cpu()
    return reward_probs, value_probs


def yield_from_queue(pool, q, device, prefetch=4):  # pylint: disable=invalid-name
    futures = [pool.apply_async(get_from_queue, args=(q, device)) for _ in range(prefetch)]
    i = 0
    while True:
        if futures[i].ready():
            future = futures[i]
            futures[i] = pool.apply_async(get_from_queue, args=(q, device))
            yield future.get()

        i = (i + 1) % prefetch
