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
import io
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


def get_from_queue(q, device):  # pylint: disable=invalid-name
    batch = q.get()
    for item in batch:
        item.to(device)
    return batch


class TrainWorker(mp.Process):
    def __init__(self, agent, batch_queue, results_queue, config):
        super().__init__()
        self.agent = agent
        self.batch_queue = batch_queue
        self.results_queue = results_queue
        self.config = config
        self.steps_per_update = config.training.model_publication_interval

    def run(self):
        asyncio.run(self.main())

    async def main(self):
        torch.set_num_threads(self.config.training.threads_per_worker)
        # original agent sent from another process, we want to work with a copy
        agent = copy.deepcopy(self.agent)
        agent.set_device(self.config.training.train_device)
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

        while True:
            lr = lr_schedule.update()  # pylint: disable=invalid-name
            epsilon = epsilon_schedule.update()
            temperature = temperature_schedule.update()
            agent.params.learning_rate = lr
            agent.params.exploration_epsilon = epsilon
            agent.params.mcts_temperature = temperature
            # batch = next(batch_generator)
            batch = get_from_queue(self.batch_queue, self.config.training.train_device)
            _priority, info = agent.learn(batch)
            del batch

            info = dict(
                lr=lr,
                epsilon=epsilon,
                temperature=temperature,
                batch_queue=self.batch_queue.qsize(),  # monitor if training process is starved
                **info,
            )

            for key, val in info.items():
                if isinstance(val, torch.Tensor):
                    info[key] = val.detach().cpu().numpy().item()

            step += 1
            if step % self.steps_per_update == 0:
                # cpu_agent = copy.deepcopy(agent)
                # cpu_agent.set_device("cpu")
                # self.results_queue.put((info, cpu_agent))
                # del cpu_agent
                # self.results_queue.put((info, agent))
                serialized_model = io.BytesIO()
                agent.save(serialized_model)
                self.results_queue.put((info, serialized_model.getvalue()))
            else:
                self.results_queue.put((info, None))