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

import copy
import queue
import torch
from cogment_verse_torch_agents.muzero.schedule import LinearScheduleWithWarmup
from cogment_verse_torch_agents.muzero.utils import MuZeroWorker, flush_queue


def get_from_queue(q, device):  # pylint: disable=invalid-name
    batch = q.get(timeout=1.0)
    for item in batch:
        item.to(device)
    return batch


class TrainWorker(MuZeroWorker):
    def __init__(self, agent, config, manager):
        super().__init__(config, manager)
        self.agent = agent
        # limit to small size so that training and sample generation don't get out of sync
        max_prefetch_batch = 128
        self.batch_queue = manager.Queue(max_prefetch_batch)
        self.results_queue = manager.Queue(max_prefetch_batch)
        self.steps_per_update = config.training.model_publication_interval

    async def main(self):
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

        while not self.done.value:
            try:
                batch = get_from_queue(self.batch_queue, self.config.train_device)
            except queue.Empty:
                continue

            lr = lr_schedule.update()  # pylint: disable=invalid-name
            epsilon = epsilon_schedule.update()
            temperature = temperature_schedule.update()
            agent.params.training.optimizer.learning_rate = lr
            agent.params.mcts.exploration_epsilon = epsilon
            agent.params.mcts.temperature = temperature

            info = agent.learn(batch)
            del batch

            info = dict(
                lr=lr,
                epsilon=epsilon,
                temperature=temperature,
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

    def cleanup(self):
        flush_queue(self.results_queue)
