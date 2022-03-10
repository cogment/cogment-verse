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
from collections import defaultdict
import ctypes
import torch
import torch.multiprocessing as mp


def flush_queue(queue):
    while not queue.empty():
        queue.get()


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


class MuZeroWorker(mp.Process):
    def __init__(self, config, manager):
        super().__init__()
        self.config = config
        self.done = manager.Value(ctypes.c_bool, False)

    def cleanup(self):
        pass

    def run(self):
        try:
            torch.set_num_threads(self.config.threads_per_worker)
            asyncio.run(self.main())
        finally:
            self.cleanup()

    def set_done(self, value):
        self.done.value = value

    async def main(self):
        raise NotImplementedError
