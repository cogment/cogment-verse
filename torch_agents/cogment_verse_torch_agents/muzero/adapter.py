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
from collections import defaultdict, namedtuple
import ctypes
import itertools
import copy
import time

from numpy.lib.type_check import real

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

from cogment_verse.utils import LRU
from cogment_verse import AgentAdapter
from cogment_verse.model_registry_client import get_model_registry_client
from cogment_verse import MlflowExperimentTracker
from cogment_verse_torch_agents.wrapper import np_array_from_proto_array, proto_array_from_np_array
from cogment_verse_torch_agents.muzero.replay_buffer import Episode, TrialReplayBuffer, EpisodeBatch

from cogment.api.common_pb2 import EnvironmentConfig, TrialState
import cogment

import logging
import torch
import numpy as np


log = logging.getLogger(__name__)

from .agent import MuZeroAgent

# pylint: disable=arguments-differ

import torch.multiprocessing as mp
from threading import Thread
import queue


MuZeroSample = namedtuple("MuZeroSample", ["state", "action", "reward", "next_state", "done", "policy", "value"])


class LinearScheduleWithWarmup:
    """Defines a linear schedule between two values over some number of steps.

    If updated more than the defined number of steps, the schedule stays at the
    end value.
    """

    def __init__(self, init_value, end_value, total_steps, warmup_steps):
        """
        Args:
            init_value (Union[int, float]): starting value for schedule.
            end_value (Union[int, float]): end value for schedule.
            steps (int): Number of steps for schedule. Should be positive.
        """
        self._warmup_steps = max(warmup_steps, 0)
        self._total_steps = max(total_steps, self._warmup_steps)
        self._init_value = init_value
        self._end_value = end_value
        self._current_step = 0
        self._value = 0

    def get_value(self):
        return self._value

    def update(self):
        if self._current_step < self._warmup_steps:
            t = np.clip(self._current_step / (self._warmup_steps + 1), 0, 1)
            self._value = self._init_value * t
        else:
            t = np.clip((self._current_step - self._warmup_steps) / (self._total_steps - self._warmup_steps), 0, 1)
            self._value = self._init_value + t * (self._end_value - self._init_value)

        self._current_step += 1
        return self._value


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
    model_publication_interval=500,
    trial_count=1000,
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
    target_label_smoothing_factor=0.01,
    target_label_smoothing_factor_steps=1,
    s_weight=1e-2,
    v_weight=0.1,
    train_device="cpu",
    actor_device="cpu",
    reanalyze_device="cpu",
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
        self._dtype = torch.float
        mp.set_start_method("spawn")

    def _create(self, model_id, *, obs_dim, act_dim, device, training_config):
        return MuZeroAgent(obs_dim=obs_dim, act_dim=act_dim, device=device, training_config=training_config)

    def _load(self, model_id, version_number, version_user_data, model_data_f):
        return MuZeroAgent.load(model_data_f, "cpu")

    def _save(self, model, model_data_f):
        assert isinstance(model, MuZeroAgent)
        model.save(model_data_f)
        return {}

    def _cached_model_key(self, model_id):
        return f"/cache/{model_id}/latest"

    def _cache_model(self, model_id, model):
        self._model_cache[self._cached_model_key(model_id)] = model

    async def _latest_model(self, model_id):
        key = self._cached_model_key(model_id)
        if key in self._model_cache:
            return self._model_cache[key]
        model, _ = await self.retrieve_version(model_id, -1)
        return model

    def _create_actor_implementations(self):
        async def _single_agent_muzero_actor_implementation(actor_session):
            actor_session.start()
            config = actor_session.config

            event_queue = mp.Queue()
            action_queue = mp.Queue()
            # event_queue = queue.Queue()
            # action_queue = queue.Queue()
            terminate = mp.Value(ctypes.c_bool, False)

            agent = await self._latest_model(config.model_id)
            agent = copy.deepcopy(agent)
            agent.set_device(config.device)
            worker = AgentTrialWorker(agent, event_queue, action_queue, terminate)

            try:
                # print("STARTING WORKER")
                worker.start()
                # print("WORKER STARTED")
                async for event in actor_session.event_loop():
                    # print("CHECKING IF WORKER IS ALIVE")
                    assert worker.is_alive()
                    # print("WORKER IS ALIVE")
                    if event.observation and event.type == cogment.EventType.ACTIVE:
                        # print("PUTTING EVENT IN QUEUE")
                        event_queue.put(event)
                        # print("GETTING ACTION FROM QUEUE")
                        actor_session.do_action(action_queue.get())
            finally:
                terminate.value = True
                # print("JOINING WORKER")
                # worker.join()
                worker.terminate()

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


def make_trial_configs(run_session, config, model_id, model_version_number):
    demonstration_env_config = copy.deepcopy(config.environment)
    demonstration_env_config.render = True
    actor_config = ActorConfig(
        model_id=model_id,
        model_version=model_version_number,
        num_input=config.actor.num_input,
        num_action=config.actor.num_action,
        env_type=config.environment.env_type,
        env_name=config.environment.env_name,
        device=config.training.actor_device,
    )
    muzero_config = TrialActor(
        name="agent_1",
        actor_class="agent",
        implementation="muzero_mlp",
        config=actor_config,
    )
    teacher_config = TrialActor(
        name="web_actor",
        actor_class="teacher_agent",
        implementation="client",
        config=actor_config,
    )
    demonstration_configs = [
        TrialConfig(
            run_id=run_session.run_id,
            environment_config=demonstration_env_config,
            actors=[muzero_config, teacher_config],
        )
        for _ in range(config.training.demonstration_trials)
    ]

    trial_configs = [
        TrialConfig(
            run_id=run_session.run_id,
            environment_config=config.environment,
            actors=[muzero_config],
        )
        for _ in range(config.training.trial_count - config.training.demonstration_trials)
    ]

    return demonstration_configs + trial_configs


async def single_agent_muzero_sample_producer_implementation(agent_adapter, run_sample_producer_session):
    # allow up to two players for human/expert intervention
    assert run_sample_producer_session.count_actors() in (1, 2)
    state = None
    step = 0
    total_reward = 0
    samples = []
    state, action, reward, policy, value = None, None, None, None, None

    async for sample in run_sample_producer_session.get_all_samples():
        observation = sample.get_actor_observation(0)
        next_state = agent_adapter.tensor_from_cog_obs(observation)
        done = sample.get_trial_state() == TrialState.ENDED
        current_player = (
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
        action, policy, value = agent_adapter.decode_cog_action(sample.get_actor_action(current_player))
        reward = sample.get_actor_reward(current_player)


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
        device=config.training.train_device,
        training_config=config.training,
    )

    episode_samples = {}
    agent_adapter._cache_model(model_id, agent)

    episode_queue = mp.Queue()
    priority_update_queue = mp.Queue()
    reanalyze_update_queue = mp.Queue()
    reanalyze_queue = mp.Queue(10)
    max_prefetch_batch = 128
    batch_queue = mp.Queue(max_prefetch_batch)  # max_prefetch_batch)  # todo: fix this?
    reanalyze_queue = mp.Queue()
    agent_update_queue = mp.Queue()
    # limit to small size so that training and sample generation don't get out of sync
    info_queue = mp.Queue(2)

    num_reanalyze_workers = 2
    agent_queue = mp.Queue()
    reanalyze_agent_queue = mp.Queue()

    train_worker = TrainWorker(agent, batch_queue, agent_update_queue, info_queue, config)
    replay_buffer = ReplayBufferWorker(
        episode_queue, priority_update_queue, batch_queue, reanalyze_queue, reanalyze_update_queue, config.training
    )
    reanalyze_agent_queue = [mp.Queue() for _ in range(num_reanalyze_workers)]
    reanalyze_workers = [
        ReanalyzeWorker(
            reanalyze_queue, reanalyze_update_queue, reanalyze_agent_queue[i], config.training.reanalyze_device
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

    workers = [train_worker, replay_buffer] + reanalyze_workers

    try:
        for worker in workers:
            worker.start()

        start_time = time.time()

        sample_generator = run_session.start_trials_and_wait_for_termination(
            trial_configs,
            max_parallel_trials=config.training.max_parallel_trials,
        )

        async for _step, timestamp, trial_id, _tick, (sample, total_reward) in sample_generator:
            for worker in workers:
                assert worker.is_alive()

            if trial_id not in episode_samples:
                episode_samples[trial_id] = Episode(sample.state, config.training.discount_rate)

            episode_samples[trial_id].add_step(
                sample.next_state, sample.action, sample.reward, sample.done, sample.policy, sample.value
            )
            total_samples += 1

            if sample.done:
                trials_completed += 1
                xp_tracker.log_metrics(
                    timestamp, total_samples, trial_total_reward=total_reward, trials_completed=trials_completed
                )
                run_total_reward += total_reward
                replay_buffer.add_episode(episode_samples.pop(trial_id))

            if replay_buffer.size() <= config.training.min_replay_buffer_size:
                continue

            # get latest online model
            try:
                agent = agent_update_queue.get_nowait()
                agent_adapter._cache_model(model_id, agent)
                cpu_agent = copy.deepcopy(agent)
                cpu_agent.set_device("cpu")
                for i, agent_queue in enumerate(reanalyze_agent_queue):
                    agent_queue.put(cpu_agent)
            except queue.Empty:
                pass

            priority, info = info_queue.get()

            training_step += 1
            info["model_version"] = version_info["version_number"]
            info["training_step"] = training_step
            info["mean_trial_reward"] = run_total_reward / max(1, trials_completed)
            info["samples_per_sec"] = total_samples / max(1, time.time() - start_time)
            running_stats.update(info)

            if training_step % config.training.model_publication_interval == 0:
                version_info = await agent_adapter.publish_version(model_id, agent)

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


class ReplayBufferWorker(mp.Process):
    def __init__(
        self, episode_queue, priority_update_queue, batch_queue, reanalyze_queue, reanalyze_update_queue, config
    ):
        super().__init__()
        self._episode_queue = episode_queue
        self._priority_update_queue = priority_update_queue
        self._batch_queue = batch_queue
        self._reanalyze_queue = reanalyze_queue
        self._reanalyze_update_queue = reanalyze_update_queue
        self._replay_buffer_size = mp.Value(ctypes.c_uint32, 0)
        self._training_config = config
        self._device = config.train_device

    def run(self):
        replay_buffer = TrialReplayBuffer(
            max_size=self._training_config.max_replay_buffer_size,
            discount_rate=self._training_config.discount_rate,
            bootstrap_steps=self._training_config.bootstrap_steps,
        )

        while True:
            # Fetch & perform all pending priority updates
            while not self._priority_update_queue.empty():
                try:
                    episodes, steps, priorities = self._priority_update_queue.get_nowait()
                    replay_buffer.update_priorities(episodes, steps, priorities)
                except queue.Empty:
                    pass

            # Add any queued data to the replay buffer
            try:
                episode = self._episode_queue.get_nowait()
                episode.bootstrap_value(self._training_config.bootstrap_steps, self._training_config.discount_rate)
                replay_buffer.add_episode(episode)
            except queue.Empty:
                pass

            self._replay_buffer_size.value = replay_buffer.size()
            if self._replay_buffer_size.value < self._training_config.min_replay_buffer_size:
                continue

            # Fetch/perform any pending reanalyze updates
            while not self._reanalyze_update_queue.empty():
                try:
                    episode_id, episode = self._reanalyze_update_queue.get_nowait()
                    replay_buffer._episodes[episode_id] = episode
                except queue.Empty:
                    pass

            # Queue next reanalyze update
            try:
                episode_id = np.random.randint(0, len(replay_buffer._episodes))
                self._reanalyze_queue.put_nowait((episode_id, replay_buffer._episodes[episode_id]))
            except queue.Full:
                pass

            # Sample a batch and add it to the training queue
            batch = replay_buffer.sample(self._training_config.rollout_length, self._training_config.batch_size)
            for item in batch:
                item.share_memory_()
            try:
                self._batch_queue.put(EpisodeBatch(*batch), timeout=1.0)
            except queue.Full:
                pass

    def add_episode(self, episode):
        self._episode_queue.put(episode)

    def get_training_batch(self):
        return self._batch_queue.get()

    def update_priorities(self, episodes, steps, priorities):
        self._priority_update_queue.put((episodes, steps, priorities))

    def size(self):
        return self._replay_buffer_size.value


class AgentTrialWorker(mp.Process):
    # class AgentTrialWorker(Thread):
    def __init__(self, agent, event_queue, action_queue, terminate):
        super().__init__()
        self._agent = agent
        self._event_queue = event_queue
        self._action_queue = action_queue
        self._terminate = terminate

    def run(self):
        while not self._terminate.value:
            try:
                event = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            obs = np_array_from_proto_array(event.observation.snapshot.vectorized)
            action, policy, value = self._agent.act(torch.tensor(obs))
            self._action_queue.put(
                AgentAction(discrete_action=action, policy=proto_array_from_np_array(policy), value=value)
            )


class TrainWorker(mp.Process):
    def __init__(self, agent, batch_queue, agent_update_queue, info_queue, config):
        super().__init__()
        self.batch_queue = batch_queue
        self.agent = agent
        self.agent_update_queue = agent_update_queue
        self.info_queue = info_queue
        self.config = config
        self.steps_per_update = 1  # 200

    def run(self):
        self.agent.set_device(self.config.training.train_device)
        step = 0

        lr_schedule = LinearScheduleWithWarmup(
            self.config.training.learning_rate,
            self.config.training.min_learning_rate,
            self.config.training.lr_decay_steps,
            self.config.training.lr_warmup_steps,
        )

        epsilon_schedule = LinearScheduleWithWarmup(
            self.config.training.exploration_epsilon,
            self.config.training.epsilon_min,
            self.config.training.epsilon_decay_steps,
            0,
        )

        temperature_schedule = LinearScheduleWithWarmup(
            self.config.training.mcts_temperature,
            self.config.training.min_temperature,
            self.config.training.temperature_decay_steps,
            0,
        )

        target_label_smoothing_schedule = LinearScheduleWithWarmup(
            1.0,
            self.config.training.target_label_smoothing_factor,
            self.config.training.target_label_smoothing_factor_steps,
            0,
        )

        while True:
            lr = lr_schedule.update()
            epsilon = epsilon_schedule.update()
            temperature = temperature_schedule.update()
            # test
            # temperature = max(0.25, temperature * 0.995)
            target_label_smoothing_factor = target_label_smoothing_schedule.update()

            self.agent._params.learning_rate = lr
            self.agent._params.exploration_epsilon = epsilon
            self.agent._params.mcts_temperature = temperature
            self.agent._params.target_label_smoothing_factor = target_label_smoothing_factor

            batch = self.batch_queue.get()
            for item in batch:
                item.to(self.config.training.train_device)

            priority, info = self.agent.learn(batch)

            info = dict(
                lr=lr,
                epsilon=epsilon,
                temperature=temperature,
                target_label_smoothing_factor=target_label_smoothing_factor,
                **info,
            )
            self.info_queue.put((priority, info))

            step += 1
            if step % self.steps_per_update == 0:
                cpu_agent = copy.deepcopy(self.agent)
                cpu_agent.set_device("cpu")
                self.agent_update_queue.put(cpu_agent)

            # for k in range(config.training.rollout_length):
            #    replay_buffer.update_priorities(batch.episode, batch.step + k, priority[:, k])


class ReanalyzeWorker(mp.Process):
    def __init__(self, reanalyze_queue, reanalyze_update_queue, agent_queue, device):
        super().__init__()
        self._reanalyze_queue = reanalyze_queue
        self._reanalyze_update_queue = reanalyze_update_queue
        self._agent_queue = agent_queue
        self._device = device

    def run(self):
        agent = self._agent_queue.get()
        agent.set_device(self._device)

        while True:
            try:
                agent = self._agent_queue.get_nowait()
                agent.set_device(self._device)
            except queue.Empty:
                pass

            episode_id, episode = self._reanalyze_queue.get()
            reanalyze_episode = Episode(episode._states[0], agent._params.discount_rate)
            for step in range(len(episode)):
                policy, _, value = agent.reanalyze(episode._states[step].clone())
                policy = policy.to("cpu")
                value = value.to("cpu").item()
                reanalyze_episode.add_step(
                    episode._states[step + 1],
                    episode._actions[step],
                    episode._rewards[step],
                    episode._done[step],
                    policy,
                    value,
                )
            episode.bootstrap_value(agent._params.bootstrap_steps, agent._params.discount_rate)
            self._reanalyze_update_queue.put((episode_id, episode))
            print("REANALYZE UPDATED UPDATE QUEUE")
