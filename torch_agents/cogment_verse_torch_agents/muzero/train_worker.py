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
import copy
import torch
import torch.multiprocessing as mp
from cogment_verse_torch_agents.muzero.schedule import LinearScheduleWithWarmup


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
        torch.set_num_threads(self.config.threads_per_worker)
        # original agent sent from another process, we want to work with a copy
        agent = copy.deepcopy(self.agent)
        agent.set_device(self.config.train_device)
        step = 0

        lr_schedule = LinearScheduleWithWarmup(
            self.config.training.optimizer.learning_rate,
            self.config.training.optimizer.min_learning_rate,
            self.config.training.optimizer.lr_decay_steps,
            self.config.training.optimizer.lr_warmup_steps,
        )

        epsilon_schedule = LinearScheduleWithWarmup(
            self.config.mcts.exploration_epsilon,
            self.config.mcts.epsilon_min,
            self.config.mcts.epsilon_decay_steps,
            0,
        )

        temperature_schedule = LinearScheduleWithWarmup(
            self.config.mcts.temperature,
            self.config.mcts.min_temperature,
            self.config.mcts.temperature_decay_steps,
            0,
        )

        while True:
            lr = lr_schedule.update()  # pylint: disable=invalid-name
            epsilon = epsilon_schedule.update()
            temperature = temperature_schedule.update()
            agent.params.training.optimizer.learning_rate = lr
            agent.params.mcts.exploration_epsilon = epsilon
            agent.params.mcts.temperature = temperature
            batch = get_from_queue(self.batch_queue, self.config.train_device)
            info = agent.learn(batch)
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
                self.results_queue.put((info, agent.serialize_to_buffer()))
            else:
                self.results_queue.put((info, None))
