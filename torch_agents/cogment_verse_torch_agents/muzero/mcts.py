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
import torch
from typing import List

# pylint: disable=invalid-name


class MinMaxInfo:
    def __init__(self, min_value=-1.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def update(self, val):
        self.min_value = min(self.min_value, val)
        self.max_value = max(self.max_value, val)

    def normalize(self, val):
        return (val - self.min_value) / max(self.max_value - self.min_value, 1)


class Node:
    def __init__(self):
        self.reward = 0.0
        self.value_sum = 0.0
        self.visit_count = 0
        self.prior = None
        self.state = None
        self.children = []

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expanded(self):
        return bool(self.children)

    def expand(self, reward, prior, state):
        self.reward = reward
        self.prior = prior
        self.state = state
        self.children = [Node() for _ in range(len(prior.view(-1)))]

    def Q(self):
        return torch.tensor([node.value() for node in self.children], device=self.prior.device)

    def N(self):
        return torch.tensor([node.visit_count for node in self.children], device=self.prior.device)

    def policy(self):
        return self.N() / self.visit_count

    def ucb(self, c1, c2, min_max_stats):
        """
        upper confidence bound
        """
        if self.visit_count == 0:
            return self.prior
        q = min_max_stats.normalize(torch.tensor(self.Q()))
        N = torch.tensor(self.N(), device=self.prior.device)
        p = self.prior
        N_sum = torch.sum(N)

        # 1911.08265 equation (2)
        return q + p * torch.sqrt(N_sum) / (1 + N) * (c1 + torch.log((N_sum + c2 + 1) / c2))


class MCTS:
    def __init__(
        self,
        *,
        policy,
        value,
        dynamics,
        state,
        discount=0.99,
        epsilon=0.1,
        alpha=1.0,
        ucb_c1=1.5,
        ucb_c2=20000.0,
    ):
        self._policy = policy
        self._value = value
        self._dynamics = dynamics
        self._max_depth = 0
        self._min_max_stats = MinMaxInfo()
        self._epsilon = epsilon
        self._alpha = alpha
        self._c1 = ucb_c1
        self._c2 = ucb_c2
        self._discount = discount

        prior = self._policy(state)
        # add exploration noise to prior at root
        concentration = torch.zeros_like(prior)
        concentration[:, :] = self._alpha
        noise = torch.distributions.Dirichlet(concentration).sample()
        prior = (1 - self._epsilon) * prior + self._epsilon * noise

        self.root = Node()
        self.root.expand(0.0, prior, state)

    def improved_targets(self, temperature):
        """
        :return: Tuple (target policy, target q, target value)
        """
        policy = self.root.policy().view(1, -1)
        q = self.root.Q().view(1, -1)
        value = torch.tensor(self.root.value().item(), device=self.root.prior.device)
        return policy, q, value

    def select_child(self, node):
        assert node.expanded()
        ucb = node.ucb(self._c1, self._c2, self._min_max_stats)
        action = torch.argmax(ucb, dim=1)
        action_int = action.cpu().numpy().item()
        return action, node.children[action_int]

    def run_simulation(self):
        node = self.root
        search_path = [self.root]

        # follow UCB policy until we reach a leaf
        while node.expanded():
            action, node = self.select_child(node)
            search_path.append(node)

        # update and expand the leaf
        parent = search_path[-2]
        value = self._value(parent.state)
        prior = self._policy(parent.state)
        state, reward = self._dynamics(parent.state, action)
        node.expand(reward, prior, state)
        self.backpropagate(search_path, value)

    def build_search_tree(self, num_simulations):
        for _ in range(num_simulations):
            self.run_simulation()

    def backpropagate(self, search_path: List[Node], value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # 1911.08265 equation (3)
            self._min_max_stats.update(node.reward + self._discount * node.value())
            # 1911.08265 equation (4)
            value = node.reward + self._discount * value
