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
import queue
import torch
import torch.multiprocessing as mp

from data_pb2 import AgentAction

from cogment_verse_torch_agents.wrapper import np_array_from_proto_array, proto_array_from_np_array
from cogment_verse_torch_agents.muzero.utils import MuZeroWorker


class AgentTrialWorker(MuZeroWorker):
    def __init__(self, agent, config, manager, sleep_time=0.01):
        super().__init__(config, manager)
        self._event_queue = mp.Queue(1)
        self._action_queue = mp.Queue(1)
        self._agent = agent
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

    async def main(self):
        while not self.done.value:
            try:
                event = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            observation = event.observation.snapshot.vectorized
            observation = np_array_from_proto_array(observation)
            action_int, policy, value = self._agent.act(torch.tensor(observation))
            action = AgentAction(discrete_action=action_int, policy=proto_array_from_np_array(policy), value=value)
            self._action_queue.put(action)
