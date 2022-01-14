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

# pylint: disable=invalid-name


class ValInfo:
    def __init__(self):
        self.vmin: float = np.inf
        self.vmax: float = -np.inf


class MCTS:
    def __init__(
        self,
        *,
        policy,
        value,
        dynamics,
        representation,
        max_depth,
        discount=0.99,
        epsilon=0.1,
        alpha=1.0,
        ucb_c1=1.5,
        ucb_c2=20000.0,
        valinfo=None,
        root=True,
    ):

        self._policy = policy
        self._value = value
        self._dynamics = dynamics
        self._prior = self._policy(representation)
        self._root = root

        self.representation = representation
        self._max_depth = max_depth
        self._valinfo = valinfo or ValInfo()
        self._epsilon = epsilon
        self._alpha = alpha
        self._c1 = ucb_c1
        self._c2 = ucb_c2

        self._discount = discount

        self._children = {}
        self._Q = torch.zeros_like(self._prior)
        self._N = torch.zeros_like(self._prior, dtype=torch.int32)
        self._R = torch.zeros_like(self._prior)

        self._cache_value = None

        if self._root:
            concentration = torch.zeros_like(self._prior)
            concentration[:, :] = self._alpha
            noise = torch.distributions.Dirichlet(concentration).sample()
            self._prior = (1 - self._epsilon) * self._prior + self._epsilon * noise

    def build_search_tree(self, count):
        for _ in range(count):
            self.rollout()

    def q_normalized(self):
        return torch.clamp(
            (self._Q - self._valinfo.vmin) / max(self._valinfo.vmax - self._valinfo.vmin, 0.01), 0.0, 1.0
        )

    def improved_targets(self, temperature):
        """
        :return: Tuple (target policy, target q, target value)
        """
        policy = self._N / torch.sum(self._N)
        policy = torch.pow(policy, 1 / temperature)
        policy /= torch.sum(policy, dim=1)
        value = torch.sum(policy * self._Q)
        return policy, self._Q, value

    def ucb(self, c1, c2):
        """
        upper confidence bound
        """
        # 1911.08265 equation (2)
        q = self.q_normalized()
        N = torch.sum(self._N)
        p = self._prior

        if N == 0:
            return p

        return q + p * torch.sqrt(N) / (1 + self._N) * (c1 + torch.log((N + c2 + 1) / c2))

    def select_child(self):
        ucb = self.ucb(self._c1, self._c2)
        action = torch.argmax(ucb, dim=1)
        action_int = action.cpu().numpy().item()

        if action_int not in self._children.keys():
            representation, reward = self._dynamics(self.representation, action)
            self._R[:, action_int] = reward
            self._children[action_int] = MCTS(
                policy=self._policy,
                value=self._value,
                dynamics=self._dynamics,
                representation=representation,
                discount=self._discount,
                max_depth=self._max_depth - 1,
                valinfo=self._valinfo,
                root=False,
            )

        return action_int, self._children[action_int]

    def update_valinfo(self, val):
        vmax = torch.max(val).detach().cpu().numpy().item()
        vmin = torch.min(val).detach().cpu().numpy().item()
        self._valinfo.vmin = min(self._valinfo.vmin, vmin)
        self._valinfo.vmax = max(self._valinfo.vmax, vmax)

    def rollout(self):
        if self._max_depth == 0:
            if self._cache_value is None:
                G = self._value(self.representation)
                self.update_valinfo(G)
                self._cache_value = G
            else:
                G = self._cache_value
            return G

        # 1911.08265 equation (3)
        action_int, child = self.select_child()
        G = self._R[:, action_int] + self._discount * child.rollout()

        # 1911.08265 equation (4)
        self._Q[:, action_int] = (self._N[:, action_int] * self._Q[:, action_int] + G) / (self._N[:, action_int] + 1)
        self._N[:, action_int] += 1

        self.update_valinfo(self._Q)
        return G
