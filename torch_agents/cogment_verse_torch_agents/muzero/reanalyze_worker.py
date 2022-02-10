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


class ReanalyzeWorker(mp.Process):
    def __init__(
        self,
        agent_queue,
        reanalyze_queue,
        reanalyze_update_queue,
        model_id,
        device,
        reward_distribution,
        value_distribution,
        max_threads,
    ):
        super().__init__()
        self._agent_queue = agent_queue
        self._reanalyze_queue = reanalyze_queue
        self._reanalyze_update_queue = reanalyze_update_queue
        self._device = device
        self._reanalyzed_samples = mp.Value(ctypes.c_uint64, 0)
        self.reward_distribution = reward_distribution
        self.value_distribution = value_distribution
        self._model_cache = LRU(1)
        self._model_id = model_id
        self._max_threads = max_threads

    def reanalyzed_samples(self):
        return self._reanalyzed_samples.value

    def update_agent(self, agent):
        self._agent_queue.put(agent)

    def run(self):
        asyncio.run(self.main())

    async def main(self):
        torch.set_num_threads(self._max_threads)
        agent = self._agent_queue.get()
        agent.set_device(self._device)

        while True:
            try:
                agent = self._agent_queue.get_nowait()
                agent.set_device(self._device)
            except queue.Empty:
                pass

            episode_id, episode = self._reanalyze_queue.get()
            reanalyze_episode = Episode(episode.states[0].clone(), agent.params.discount_rate)
            for step in range(len(episode)):
                policy, _, value = agent.reanalyze(episode.states[step].clone())
                policy = policy.cpu()
                value = value.cpu().item()
                reward_probs = self.reward_distribution.compute_target(torch.tensor(episode.rewards[step])).cpu()
                value_probs = self.value_distribution.compute_target(torch.tensor(value)).cpu()
                reanalyze_episode.add_step(
                    episode.states[step + 1].clone(),
                    episode.actions[step],
                    reward_probs.clone(),
                    episode.rewards[step],
                    episode.done[step],
                    policy.clone(),
                    value_probs.clone(),
                    value,
                )
            episode.bootstrap_value(agent.params.bootstrap_steps, agent.params.discount_rate)
            self._reanalyze_update_queue.put((episode_id, episode))
            self._reanalyzed_samples.value += len(episode)
