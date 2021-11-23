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

from data_pb2 import Coordinate

from math import pi


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

    def __repr__(self):
        return repr((self.position[1], self.position[0]))

    def add_pipe(self, pipe):
        self.pipes.append(pipe)

    def is_directly_linked_to(self, node):
        for pipe in self.pipes:
            if node.node_id == pipe.node1.node_id or node.node_id == pipe.node2.node_id:
                return True
        return False

    def step_water_flow(self):
        pipe_count = len(self.pipes)
        if pipe_count > 0 and self.flow != 0.0:
            left_water = 0.0
            water_by_pipe = self.flow / pipe_count
            for pipe in self.pipes:
                left_water = pipe.add_water(self, water_by_pipe + left_water)

            if left_water != 0.0:
                for pipe in self.pipes:
                    left_water = pipe.add_water(self, left_water)

    def step_balance_water(self):
        pipe_count = len(self.pipes)
        if pipe_count > 0:
            total_water = 0.0
            total_capacity = 0.0

            for pipe in self.pipes:
                total_water += pipe.water
                total_capacity += pipe.capacity_empty

            water_level = min(1.0, total_water / total_capacity)

            check_sum = 0.0
            for pipe in self.pipes:
                water = water_level * pipe.capacity_empty
                pipe.set_water(self, water)
                check_sum += water

            if abs(check_sum - total_water) > 1.0:
                print("Warning some water have been loose in the simulator")


class Pipe:
    def __init__(self, pipe_id, node1, node2):
        self.pipe_id = pipe_id
        self.node1 = node1
        self.node2 = node2
        self.node1.add_pipe(self)
        self.node2.add_pipe(self)
        self.water = 0.0
        self.water_added_by_node1 = 0.0
        self.water_added_by_node2 = 0.0
        self.pressure = 0.0
        self.condition = 9.0 + random.random()
        self.prob_of_failure = 0.0
        self.is_failing = False
        self.estimate_error = random.random() - 0.5
        self.cost_of_failure = 0.0
        diff = node1.position - node2.position
        self.length = np.linalg.norm(diff)
        self.radius = 0.5
        self.area = pi * self.radius * self.radius
        self.capacity_empty = self.length * self.area
        self.added_in_logical = False
        self.inspected = False


    def clear_water_flow(self):
        self.water_added_by_node1 = 0.0
        self.water_added_by_node2 = 0.0

    def inspect(self):
        self.inspected = True

    def return_cost_of_failure(self):
        if self.inspected:
            return self.cost_of_failure
        else:
            return self.cost_of_failure + self.estimate_error * self.cost_of_failure

    def add_water(self, node, water):
        if node == self.node1:
            self.water_added_by_node1 += water
        elif node == self.node2:
            self.water_added_by_node2 += water

        if self.is_failing:
            self.water = 0.0
            return 0.0
        if self.water + water <= self.capacity_empty:
            self.water += water
            self.water = max(0.0, self.water)
            return 0.0
        else:
            left_water = water + self.water - self.capacity_empty
            self.water = self.capacity_empty
        return left_water

    def set_water(self, node, water):
        previous_water = self.water 
        if self.is_failing:
            self.water = 0.0
        else:
            self.water = min(water, self.capacity_empty)

            if node == self.node1:
                self.water_added_by_node1 += (self.water - previous_water)
            elif node == self.node2:
                self.water_added_by_node2 += (self.water - previous_water)

    def step(self):
        self.pressure = self.water / self.capacity_empty
        self.condition -= self.pressure * 0.1
        self.prob_of_failure = (10.0 - self.condition) * 0.1 * self.pressure

        # Add some factor to make the probability making sense
        if random.random() < self.prob_of_failure * 0.001:
            self.is_failing = True

    def maintain(self):
        self.is_failing = False
        self.condition = 10.0

    def encode(self, proto):
        proto.start.x = self.node1.position[0]
        proto.start.y = self.node1.position[1]
        proto.end.x = self.node2.position[0]
        proto.end.y = self.node2.position[1]
        proto.water = self.water / self.capacity_empty
        proto.water_flow = (self.water_added_by_node2 - self.water_added_by_node1) / self.capacity_empty


class City:
    def __init__(self, city_size, district_side_count=8):
        self.city_size = city_size
        self.district_side_count = district_side_count
        self.cost_of_failures = []
        self.min_cost = 1
        self.max_cost = 1000
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
    def __init__(self, city_size=1024):
        self.nodes = []
        self.pipes = []
        self.city_size = city_size
        self.city = City(city_size)
        self.city.generate_city()

        self.houses = []
        self.water_towers = []

    def generate_node(self):
        consumer_count = 20
        provider_count = 5
        no_consumer_count = 200

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
            house = Coordinate(x=node.position[0], y=node.position[1])
            self.houses.append(house)

        for _ in range(provider_count):
            node = Node(node_id)
            node_id += 1
            node.flow = 30.0 + random.random() * 100.0
            node.position = np.random.randint(0, self.city_size, 2)
            self.nodes.append(node)
            water_tower = Coordinate(x=node.position[0], y=node.position[1])
            self.water_towers.append(water_tower)

    def generate_pipeline(self):
        
        self.generate_node()

        pipe_count = 400

        node_count = len(self.nodes)
        # Use some heuristic here
        node_to_compare_count = node_count // 5
        max_length_pipe = 0.8 * self.city.district_size
        sqr_max_length_pipe = max_length_pipe * max_length_pipe
        pipe_id = 0

        loop_count = 0

        while len(self.pipes) < pipe_count:

            if (loop_count % 2 == 0):
                self.nodes.sort()
            else:
                self.nodes = sorted(self.nodes, key=lambda node: node.position[1])
            loop_count += 1



            for index, node1 in enumerate(self.nodes):
            # for _ in range(len(self.nodes)):

                # index = random.randint(0, len(self.nodes)-1)
                node1 = self.nodes[index]
                if len(node1.pipes) >= loop_count+1:
                    continue

                closest_node = None
                closest_sqr_dist = 2 * self.city_size * self.city_size

                compare_count = min(index + 1 + node_to_compare_count, len(self.nodes))

                # Heuristic not to generate huge pipes at the end of the sorted list
                # Commented handled differently
                # if index + 3 > len(self.nodes):
                #     break

                start_index = max(0, index - compare_count)
                for i in range(start_index, compare_count):
                    if i == index:
                        continue
                    node2 = self.nodes[i]
                    diff = node1.position - node2.position
                    sqr_dist = np.dot(diff, diff)
                    if (
                        sqr_dist < closest_sqr_dist
                        and not node1.is_directly_linked_to(node2)
                        and sqr_dist < sqr_max_length_pipe
                        and len(node2.pipes) < loop_count + 1
                    ):
                        closest_sqr_dist = sqr_dist
                        closest_node = node2
                if closest_node is not None:
                    pipe = Pipe(pipe_id, node1, closest_node)
                    pipe_id += 1
                    self.pipes.append(pipe)
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
        for pipe in self.pipes:
            pipe.clear_water_flow()

        for node in self.nodes:
            node.step_water_flow()

        for _ in range(1):
            for node in self.nodes:
                node.step_balance_water()

        for pipe in self.pipes:
            pipe.step()

    def encode(self, proto):
        for house in self.houses:
            h = proto.houses.add()
            h.CopyFrom(house)

        for water_tower in self.water_towers:
            t = proto.water_towers.add()
            t.CopyFrom(water_tower)

def main():

    pipeline = Pipeline()
    pipeline.generate_pipeline()

    pipeline.step()


if __name__ == "__main__":
    main()
