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
from multiprocessing.pool import ThreadPool
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
    ActorConfig,
)

from cogment_verse.utils import LRU
from cogment_verse import AgentAdapter
from cogment_verse import MlflowExperimentTracker
from cogment_verse_torch_agents.wrapper import np_array_from_proto_array, proto_array_from_np_array
from cogment_verse_torch_agents.muzero.replay_buffer import Episode, TrialReplayBuffer, EpisodeBatch
from cogment_verse_torch_agents.muzero.agent import MuZeroAgent
from cogment_verse_torch_agents.muzero.schedule import LinearScheduleWithWarmup
from cogment_verse_torch_agents.muzero.stats import RunningStats
from cogment_verse.model_registry_client import get_model_registry_client

from cogment.api.common_pb2 import TrialState
import cogment


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

    def _create(self, model_id, *, obs_dim, act_dim, device, training_config):
        return MuZeroAgent(obs_dim=obs_dim, act_dim=act_dim, device=device, training_config=training_config)

    def _load(self, model_id, version_number, version_user_data, model_data_f):
        return MuZeroAgent.load(model_data_f, "cpu")

    def _save(self, model, model_data_f):
        assert isinstance(model, MuZeroAgent)
        model.save(model_data_f)
        return {}

    def _create_actor_implementations(self):
        async def _single_agent_muzero_actor_implementation(actor_session):
            actor_session.start()
            agent, _ = await self.retrieve_version(actor_session.config.model_id, -1)
            agent = copy.deepcopy(agent)
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

            if action < 0:
                print("WARNING: override action is invalid, ignoring!")
                action = self.decode_cog_action(sample.get_actor_action(observation.current_player))

            assert action >= 0

            policy, value = self.decode_cog_policy_value(sample.get_actor_action(observation.current_player))
            reward = sample.get_actor_reward(player_override)

    async def single_agent_muzero_run_implementation(self, run_session):
        xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

        # Initializing a model
        model_id = f"{run_session.run_id}_model"

        config = run_session.config
        assert config.environment.config.player_count == 1

        agent, version_info = await self.create_and_publish_initial_version(
            model_id=model_id,
            obs_dim=config.actor.num_input,
            act_dim=config.actor.num_action,
            device=config.training.train_device,
            training_config=config.training,
        )

        num_reanalyze_workers = config.training.reanalyze_workers

        max_prefetch_batch = 128
        sample_queue = mp.Queue()
        priority_update_queue = mp.Queue()
        reanalyze_update_queue = mp.Queue()
        reanalyze_queue = mp.Queue(num_reanalyze_workers + 1)

        batch_queue = mp.Queue(max_prefetch_batch)
        reanalyze_queue = mp.Queue()
        # limit to small size so that training and sample generation don't get out of sync
        info_queue = mp.Queue(max_prefetch_batch)

        reward_distribution = copy.deepcopy(agent.muzero.reward_distribution).cpu()
        value_distribution = copy.deepcopy(agent.muzero.value_distribution).cpu()

        train_worker = TrainWorker(agent, batch_queue, info_queue, config)
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
                reanalyze_queue,
                reanalyze_update_queue,
                model_id,
                config.training.reanalyze_device,
                reward_distribution,
                value_distribution,
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

                while not info_queue.empty():
                    try:
                        info, agent_update = info_queue.get_nowait()
                        if agent_update is not None:
                            agent = agent_update
                    except queue.Empty:
                        continue

                    training_step += 1
                    info["model_version"] = version_info["version_number"]
                    info["training_step"] = training_step
                    info["mean_trial_reward"] = run_total_reward / max(1, trials_completed)
                    info["samples_per_sec"] = total_samples / max(1, time.time() - start_time)
                    info["reanalyzed_samples"] = sum([worker.reanalyzed_samples() for worker in reanalyze_workers])
                    running_stats.update(info)

                if total_samples % config.training.model_publication_interval == 0:
                    version_info = await self.publish_version(model_id, agent)

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
                        implementation="gym/CartPole-v0",
                        config=EnvironmentConfig(
                            seed=12,
                            player_count=1,
                            framestack=1,
                            render=False,
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

    env_type, env_name = config.environment.implementation.split("/")

    actor_config = ActorConfig(
        model_id=model_id,
        model_version=model_version_number,
        num_input=config.actor.num_input,
        num_action=config.actor.num_action,
        env_type=env_type,
        env_name=env_name,
        device=config.training.actor_device,
    )
    muzero_config = ActorParams(
        name="agent_1",
        actor_class="agent",
        implementation="muzero_mlp",
        config=actor_config,
    )
    teacher_config = ActorParams(
        name="web_actor",
        actor_class="teacher_agent",
        implementation="client",
        config=actor_config,
    )
    demonstration_configs = [
        TrialConfig(
            run_id=run_session.run_id,
            environment=EnvironmentParams(
                implementation=config.environment.implementation,
                config=clone_config(
                    config.environment.config,
                    seed=config.environment.config.seed + i + config.training.demonstration_trials,
                    render=False,
                ),
            ),
            actors=[muzero_config, teacher_config],
        )
        for i in range(config.training.demonstration_trials)
    ]

    trial_configs = [
        TrialConfig(
            run_id=run_session.run_id,
            environment=EnvironmentParams(
                implementation=config.environment.implementation,
                config=clone_config(
                    config.environment.config,
                    seed=config.environment.config.seed + i + config.training.demonstration_trials,
                    render=False,
                ),
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


class ReplayBufferWorker(mp.Process):
    def __init__(
        self,
        sample_queue,
        priority_update_queue,
        batch_queue,
        reanalyze_queue,
        reanalyze_update_queue,
        config,
        reward_distribution,
        value_distribution,
    ):
        super().__init__()
        self._sample_queue = sample_queue
        self._priority_update_queue = priority_update_queue
        self._batch_queue = batch_queue
        self._reanalyze_queue = reanalyze_queue
        self._reanalyze_update_queue = reanalyze_update_queue
        self._replay_buffer_size = mp.Value(ctypes.c_uint32, 0)
        self._training_config = config
        self._device = config.train_device
        self.reward_distribution = reward_distribution
        self.value_distribution = value_distribution

    def run(self):
        episode_samples = {}
        replay_buffer = TrialReplayBuffer(
            max_size=self._training_config.max_replay_buffer_size,
            discount_rate=self._training_config.discount_rate,
            bootstrap_steps=self._training_config.bootstrap_steps,
        )

        zero_reward_probs = self.reward_distribution.compute_target(torch.tensor(0.0)).cpu().detach()
        zero_value_probs = self.value_distribution.compute_target(torch.tensor(0.0)).cpu().detach()

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
                trial_id, sample = self._sample_queue.get_nowait()

                if trial_id not in episode_samples:
                    episode_samples[trial_id] = Episode(
                        sample.state,
                        self._training_config.discount_rate,
                        zero_reward_probs=zero_reward_probs,
                        zero_value_probs=zero_value_probs,
                    )

                with torch.no_grad():
                    reward_probs = self.reward_distribution.compute_target(torch.tensor(sample.reward)).cpu()
                    value_probs = self.value_distribution.compute_target(torch.tensor(sample.value)).cpu()

                episode_samples[trial_id].add_step(
                    sample.next_state,
                    sample.action,
                    reward_probs,
                    sample.reward,
                    sample.done,
                    sample.policy,
                    value_probs,
                    sample.value,
                )

                if sample.done:
                    episode_samples[trial_id].bootstrap_value(
                        self._training_config.bootstrap_steps, self._training_config.discount_rate
                    )
                    replay_buffer.add_episode(episode_samples.pop(trial_id))
            except queue.Empty:
                pass

            self._replay_buffer_size.value = replay_buffer.size()
            if self._replay_buffer_size.value < self._training_config.min_replay_buffer_size:
                continue

            # Fetch/perform any pending reanalyze updates
            try:
                episode_id, episode = self._reanalyze_update_queue.get_nowait()
                replay_buffer.episodes[episode_id] = episode
            except queue.Empty:
                pass

            # Queue next reanalyze update
            try:
                episode_id = np.random.randint(0, len(replay_buffer.episodes))
                self._reanalyze_queue.put_nowait((episode_id, replay_buffer.episodes[episode_id]))
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

    def add_sample(self, trial_id, sample):
        self._sample_queue.put((trial_id, sample))

    def update_priorities(self, episodes, steps, priorities):
        self._priority_update_queue.put((episodes, steps, priorities))

    def size(self):
        return self._replay_buffer_size.value


def get_from_queue(q, device):  # pylint: disable=invalid-name
    batch = q.get()
    for item in batch:
        item.to(device)
    return batch


def yield_from_queue(pool, q, device, prefetch=4):  # pylint: disable=invalid-name
    futures = [pool.apply_async(get_from_queue, args=(q, device)) for _ in range(prefetch)]
    i = 0
    while True:
        if futures[i].ready():
            future = futures[i]
            futures[i] = pool.apply_async(get_from_queue, args=(q, device))
            yield future.get()

        i = (i + 1) % prefetch


class AgentTrialWorker(mp.Process):
    def __init__(self, agent, config, sleep_time=0.01):
        super().__init__()
        self._event_queue = mp.Queue(1)
        self._action_queue = mp.Queue(1)
        self._config = config
        self._agent = agent
        self._terminate = mp.Value(ctypes.c_bool, False)
        self._sleep_time = sleep_time

    async def put_event(self, event):
        while True:
            try:
                self._event_queue.put_nowait(event)
                break
            except queue.Full:
                await asyncio.sleep(self._sleep_time)

    async def get_action(self):
        while True:
            try:
                return self._action_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(self._sleep_time)

    def run(self):
        while not self._terminate.value:
            try:
                event = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            observation = event.observation.snapshot.vectorized
            observation = np_array_from_proto_array(observation)
            action_int, policy, value = self._agent.act(torch.tensor(observation))
            action = AgentAction(discrete_action=action_int, policy=proto_array_from_np_array(policy), value=value)
            self._action_queue.put(action)


class TrainWorker(mp.Process):
    def __init__(self, agent, batch_queue, info_queue, config):
        super().__init__()
        self.batch_queue = batch_queue
        self.agent = agent
        self.info_queue = info_queue
        self.config = config
        self.steps_per_update = 200

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

        threadpool = ThreadPool()
        batch_generator = yield_from_queue(threadpool, self.batch_queue, self.config.training.train_device)

        while True:
            lr = lr_schedule.update()  # pylint: disable=invalid-name
            epsilon = epsilon_schedule.update()
            temperature = temperature_schedule.update()
            self.agent.params.learning_rate = lr
            self.agent.params.exploration_epsilon = epsilon
            self.agent.params.mcts_temperature = temperature
            batch = next(batch_generator)
            _priority, info = self.agent.learn(batch)

            info = dict(
                lr=lr,
                epsilon=epsilon,
                temperature=temperature,
                **info,
            )

            step += 1
            if step % self.steps_per_update == 0:
                cpu_agent = copy.deepcopy(self.agent)
                cpu_agent.set_device("cpu")
                self.info_queue.put((info, cpu_agent))
            else:
                self.info_queue.put((info, None))


class ReanalyzeWorker(mp.Process):
    def __init__(
        self, reanalyze_queue, reanalyze_update_queue, model_id, device, reward_distribution, value_distribution
    ):
        super().__init__()
        self._reanalyze_queue = reanalyze_queue
        self._reanalyze_update_queue = reanalyze_update_queue
        self._device = device
        self._reanalyzed_samples = mp.Value(ctypes.c_uint64, 0)
        self.reward_distribution = reward_distribution
        self.value_distribution = value_distribution
        self._model_cache = LRU(1)
        self._model_id = model_id

    async def get_latest_agent(self):
        def load_agent(model_id, version_number, version_user_data, model_data_f):
            return MuZeroAgent.load(model_data_f, "cpu")

        return await get_model_registry_client().retrieve_model_version(
            self._model_id, load_agent, self._model_cache, version_number=-1
        )

    def reanalyzed_samples(self):
        return self._reanalyzed_samples.value

    def run(self):
        asyncio.run(self.main())

    async def main(self):
        while True:
            agent, _ = await self.get_latest_agent()
            agent.set_device(self._device)

            episode_id, episode = self._reanalyze_queue.get()
            reanalyze_episode = Episode(episode.states[0], agent.params.discount_rate)
            for step in range(len(episode)):
                policy, _, value = agent.reanalyze(episode.states[step].clone())
                policy = policy.cpu()
                value = value.cpu().item()
                reward_probs = self.reward_distribution.compute_target(torch.tensor(episode.rewards[step]))
                value_probs = self.value_distribution.compute_target(torch.tensor(value))
                reanalyze_episode.add_step(
                    episode.states[step + 1],
                    episode.actions[step],
                    reward_probs,
                    episode.rewards[step],
                    episode.done[step],
                    policy,
                    value_probs,
                    value,
                )
            episode.bootstrap_value(agent.params.bootstrap_steps, agent.params.discount_rate)
            self._reanalyze_update_queue.put((episode_id, episode))
            self._reanalyzed_samples.value += len(episode)
