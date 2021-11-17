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

import numpy as np
from pipe_world.physical_segment import Pipeline


class LogicalSegment:
    def __init__(self):
        self.pipes = []
        self.cost_of_maintnance = 0.0
        self.cost_of_inspection = 0.0
        self.prob_of_failure = 0.0
        self.cost_of_failure = 0.0
        self.length = 0.0
        self.max_pressure = 0.0
        self.condition = 10.0
        self.is_failing = False
        self.is_inspected = False
        self.water = 0.0

    def add_pipe(self, pipe):
        self.pipes.append(pipe)
        pipe.added_in_logical = True

    def compute_info(self):
        self.cost_of_failure = 0.0
        self.prob_of_failure = 0.0
        self.condition = 10.0
        self.max_pressure = 0.0
        self.prob_of_failure = 0.0
        self.length = 0.0
        self.is_failing = False
        self.is_inspected = False
        self.water = 0.0
        for pipe in self.pipes:
            self.length += pipe.length
            self.cost_of_failure += pipe.return_cost_of_failure()
            self.condition = min(self.condition, pipe.condition)
            self.max_pressure = max(self.max_pressure, pipe.pressure)
            self.prob_of_failure = max(self.prob_of_failure, pipe.prob_of_failure)
            self.is_failing = self.is_failing or pipe.is_failing
            self.is_inspected = self.is_inspected or pipe.inspected
            self.water += pipe.water

        self.cost_of_maintnance = 100.0 + self.length * 10.0
        self.cost_of_inspection = 10.0 + self.length * 1.0

    def step(self):
        self.compute_info()

    def maintain(self):
        for pipe in self.pipes:
            pipe.maintain()

    def inspect(self):
        for pipe in self.pipes:
            pipe.inspect()

    def display(self):
        print(
            "Max P:  {:3.2f}".format(self.max_pressure),
            " Cond:  {:3.2f}".format(self.condition),
            " Prob fail:  {:3.4f}".format(self.prob_of_failure),
            " Cost F:  {:3.2f}".format(self.cost_of_failure),
            " length:  {:3.2f}".format(self.length),
            " Failure:  ",
            self.is_failing,
            " Cost M:  {:3.2f}".format(self.cost_of_maintnance),
            " Cost I:  {:3.2f}".format(self.cost_of_inspection),
            " Water:  {:3.2f}".format(self.water),
        )

    def generate_observation(self):
        is_failing = 0.0
        if self.is_failing:
            is_failing = 1.0
        is_inspected = 0.0
        if self.is_inspected:
            is_inspected = 1.0
        return np.array([self.prob_of_failure, self.cost_of_failure, is_failing, is_inspected])


class LogicalSegments:
    def __init__(self, expected_segment_count):
        self.pipeline = Pipeline()
        self.pipeline.generate_pipeline()
        self.scale = 0.0
        self.expected_segment_count = expected_segment_count
        self.logical_segments = []
        self.apply_logical_segment_count()

    def step(self):
        self.pipeline.step()
        for logical_segment in self.logical_segments:
            logical_segment.step()

        cost = 0.0
        for pipe in self.pipeline.pipes:
            if pipe.is_failing:
                cost += pipe.cost_of_failure
        return cost

    def scale_segments(self, scale):
        if self.scale != scale:
            self.scale = scale
            self.apply_scale()

    def apply_scale(self):
        assert self.scale >= 0.0
        logical_segment_length = self.scale * 400.0
        self.compute_logical_segment(logical_segment_length)


    def apply_logical_segment_count(self):
        total_length = 0.0

        for pipe in self.pipeline.pipes:
            total_length += pipe.length

        logical_segment_length = total_length / self.expected_segment_count
        self.compute_logical_segment(logical_segment_length)

    def compute_logical_segment(self, logical_segment_length):
        self.logical_segments.clear()
        self.pipeline.clear_pipes_from_logic()
        all_in_logic = False
        while not all_in_logic:
            pipe = None
            all_in_logic = True
            for other_pipe in self.pipeline.pipes:
                if other_pipe.added_in_logical is False:
                    pipe = other_pipe
                    all_in_logic = False
                    break
            if all_in_logic:
                break

            logical_segment = LogicalSegment()
            logical_segment.add_pipe(pipe)
            current_length = pipe.length
            dead_end = False
            while current_length <= logical_segment_length and not dead_end:
                dead_end = True
                for ohter_pipe in pipe.node1.pipes + pipe.node2.pipes:
                    if ohter_pipe.added_in_logical is False:
                        pipe = ohter_pipe
                        dead_end = False
                        break
                logical_segment.add_pipe(pipe)
                current_length += pipe.length
            logical_segment.compute_info()
            self.logical_segments.append(logical_segment)

    def display(self):
        for index, segment in enumerate(self.logical_segments):
            print(index, ": ", end="")
            segment.display()

    def generate_observation(self):
        info_count = 4
        observation = np.zeros((self.expected_segment_count, info_count))
        for index, segment in enumerate(self.logical_segments):
            if index >= self.expected_segment_count:
                break
            observation[index] = segment.generate_observation()
        observation = np.reshape(observation, (self.expected_segment_count*info_count))
        return observation
