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
import random


class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.pipes = []
        self.flow = 0.0
        self.position = np.array([0.0, 0.0])

    def __lt__(self, other):

        if self.position[0] == other.position[0]:
            return self.position[1] < other.position[1]

        return self.position[0] < other.position[0]

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def is_directly_linked_to(self, node):
        for pipe in self.pipes:
            if node.node_id == pipe.node1.node_id or node.node_id == pipe.node2.node_id:
                return True
        return False

    def step_water_flow(self):
        pipe_count = len(self.pipes)
        left_water = 0.0
        if pipe_count > 0 and self.flow != 0.0:
            water_by_pipe = self.flow / pipe_count
            for pipe in self.pipes:
                left_water = pipe.add_water(water_by_pipe + left_water)

    def step_balance_water(self):
        pipe_count = len(self.pipes)
        if pipe_count > 0:
            total_water = 0.0
            for pipe in self.pipes:
                total_water += pipe.water
            water_by_pipe = total_water / pipe_count

            for pipe in self.pipes:
                pipe.set_water(water_by_pipe)


class Pipe:
    def __init__(self, pipe_id, node1, node2):
        self.pipe_id = pipe_id
        self.node1 = node1
        self.node2 = node2
        self.node1.add_pipe(self)
        self.node2.add_pipe(self)
        self.water = 0.0
        self.capacity = 10.0
        self.pressure = 0.0
        self.condition = 9.0 + random.random()
        self.prob_of_failure = 0.0
        self.is_failing = False
        self.estimate_error = random.random() - 0.5
        self.cost_of_failure = 0.0
        diff = node1.position - node2.position
        self.length = np.linalg.norm(diff)
        self.added_in_logical = False
        self.inspected = False

    def inspect(self):
        self.inspected = True

    def return_cost_of_failure(self):
        if self.inspected:
            return self.cost_of_failure
        else:
            return self.cost_of_failure + self.estimate_error * self.cost_of_failure

    def add_water(self, water):
        if self.is_failing:
            self.water = 0.0
            return 0.0
        if self.water + water <= self.capacity:
            self.water += water
            self.water = max(0.0, self.water)
            return 0.0
        else:
            left_water = water + self.water - self.capacity
            self.water = self.capacity
        return left_water

    def set_water(self, water):
        if self.is_failing:
            self.water = 0.0
        else:
            self.water = water

    def step(self):
        self.pressure = self.water / self.capacity
        self.condition -= self.pressure * 0.1
        self.prob_of_failure = (10.0 - self.condition) * 0.01 * self.pressure

        if random.random() < self.prob_of_failure * 0.1:
            self.is_failing = True

    def maintain(self):
        self.is_failing = False
        self.condition = 10.0


class City:
    def __init__(self, city_size, district_side_count=8):
        self.city_size = city_size
        self.district_side_count = district_side_count
        self.cost_of_failures = []
        self.min_cost = 100
        self.max_cost = 10000
        self.district_size = 0

    def generate_city(self):
        self.district_size = self.city_size / self.district_side_count
        for _ in range(self.district_side_count * self.district_side_count):
            cost_of_failure = random.randint(self.min_cost, self.max_cost)
            self.cost_of_failures.append(cost_of_failure)

    def fill_cost_of_failures(self, pipes):
        for pipe in pipes:
            x1 = pipe.node1.position[0] // self.district_size
            y1 = pipe.node1.position[1] // self.district_size

            x2 = pipe.node2.position[0] // self.district_size
            y2 = pipe.node2.position[1] // self.district_size

            cost1 = self.cost_of_failures[int(x1 + y1 * self.district_side_count)]
            cost2 = self.cost_of_failures[int(x2 + y2 * self.district_side_count)]

            cost = (cost1 + cost2) / 2.0

            pipe.cost_of_failure = cost


class Pipeline:
    def __init__(self, city_size=256):
        self.nodes = []
        self.pipes = []
        self.city_size = city_size
        self.city = City(city_size)
        self.city.generate_city()

    def generate_pipeline(self):
        consumer_count = 20
        provider_count = 5
        no_consumer_count = 30
        pipe_count = 100

        node_id = 0
        for _ in range(no_consumer_count):
            node = Node(node_id)
            node_id += 1
            node.position = np.random.randint(0, self.city_size, 2)
            self.nodes.append(node)

        for _ in range(consumer_count):
            node = Node(node_id)
            node_id += 1
            node.flow = -random.random() * 3.0
            node.position = np.random.randint(0, self.city_size, 2)
            self.nodes.append(node)

        for _ in range(provider_count):
            node = Node(node_id)
            node_id += 1
            node.flow = 30.0 + random.random() * 100.0
            node.position = np.random.randint(0, self.city_size, 2)
            self.nodes.append(node)

        self.nodes.sort()

        node_count = len(self.nodes)
        # Use some heuristic here
        node_to_compare_count = node_count // 5
        max_length_pipe = 0.8 * self.city.district_size
        sqr_max_length_pipe = max_length_pipe * max_length_pipe

        pipe_id = 0
        while len(self.pipes) < pipe_count:
            for index, node1 in enumerate(self.nodes):
                closest_node = None
                closest_sqr_dist = 2 * self.city_size * self.city_size

                compare_count = min(index + 1 + node_to_compare_count, len(self.nodes))

                # Heuristic not to generate huge pipes at the end of the sorted list
                # Commented handled differently
                # if index + 7 > len(self.nodes):
                #     break

                for i in range(index + 1, compare_count):
                    node2 = self.nodes[i]
                    diff = node1.position - node2.position
                    sqr_dist = np.dot(diff, diff)
                    if (
                        sqr_dist < closest_sqr_dist
                        and not node1.is_directly_linked_to(node2)
                        and sqr_dist < sqr_max_length_pipe
                    ):
                        closest_sqr_dist = sqr_dist
                        closest_node = node2
                if closest_node is not None:
                    pipe = Pipe(pipe_id, node1, closest_node)
                    pipe_id += 1
                    self.pipes.append(pipe)
                    # print("Pipe ", len(self.pipes), " connect ", pipe.node1.node_id, " to ", pipe.node2.node_id)
                    # print("Pipe ", len(self.pipes), " connect ", pipe.node1.position, " to ", pipe.node2.position)
                else:
                    # Failed to connect a pipe
                    node_to_compare_count += 1

                if len(self.pipes) >= pipe_count:
                    break
                else:
                    sqr_max_length_pipe *= 1.1

        # Check connectivity
        for node in self.nodes:
            if len(node.pipes) <= 0:
                print("WARNING: node ", node.node_id, " is not connected.")

        self.city.fill_cost_of_failures(self.pipes)

    def clear_pipes_from_logic(self):
        for pipe in self.pipes:
            pipe.added_in_logical = False

    def step(self):
        for node in self.nodes:
            node.step_water_flow()

        for _ in range(1):
            for node in self.nodes:
                node.step_balance_water()

        for pipe in self.pipes:
            pipe.step()


def main():

    pipeline = Pipeline()
    pipeline.generate_pipeline()

    pipeline.step()


if __name__ == "__main__":
    main()
