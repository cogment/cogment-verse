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

import time

from cogment_verse.utils import LRU


def compute_full_tick_id(trial_id, tick_id):
    return f"{trial_id}-#{tick_id}"


class RunStepper:
    def __init__(self):
        self._steps_count = 0
        self._step_from_full_tick_id = LRU(100000)

    def count_steps(self):
        return self._steps_count

    def get_step(self, trial_id, tick_id):
        full_tick_id = compute_full_tick_id(trial_id, tick_id)
        if full_tick_id not in self._step_from_full_tick_id:
            # The step has not been "started" yet, or is no longer in cache
            raise Exception(f"Unknown step for trial [{trial_id}] at tick [{tick_id}]")
        return self._step_from_full_tick_id[full_tick_id]

    def step(self, trial_id, tick_id):
        full_tick_id = compute_full_tick_id(trial_id, tick_id)
        if full_tick_id in self._step_from_full_tick_id:
            # The step has already been "started"
            raise Exception(f"Existing step for trial [{trial_id}] at tick [{tick_id}]")

        self._steps_count += 1
        step = (self._steps_count, int(time.time() * 1000))
        self._step_from_full_tick_id[full_tick_id] = step

        return step
