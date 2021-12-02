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

from data_pb2 import MuZeroTrainingConfig

import logging
import torch
import numpy as np
import copy
import itertools

log = logging.getLogger(__name__)

from .networks import (
    MuZero,
    lin_bn_act,
    reward_transform,
    reward_transform_inverse,
    Distributional,
    DynamicsAdapter,
    resnet,
    mlp,
    RepresentationNetwork,
    PolicyNetwork,
    ValueNetwork,
    DynamicsNetwork,
    QNetwork,
)
from .replay_buffer import EpisodeBatch

# pylint: disable=arguments-differ


class MuZeroAgent:
    """
    MuZero implementation
    """

    def __init__(self, *, obs_dim, act_dim, device, training_config: MuZeroTrainingConfig):
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._params = training_config
        self._device = torch.device(device)
        self._make_networks()

    def set_device(self, device):
        self._device = torch.device(device)
        self._muzero = self._muzero.to(self._device)
        self._target_muzero = self._target_muzero.to(self._device)

    def _make_networks(self):
        stem = lin_bn_act(self._obs_dim, self._params.hidden_dim, bn=True, act=torch.nn.ReLU())
        representation = RepresentationNetwork(stem, self._params.hidden_dim, self._params.hidden_layers)
        # representation = stem

        policy = PolicyNetwork(self._params.hidden_dim, self._params.hidden_layers, self._act_dim)

        value = ValueNetwork(
            self._params.hidden_dim,
            self._params.hidden_layers,
            self._params.vmin,
            self._params.vmax,
            self._params.vbins,
        )

        dynamics = DynamicsNetwork(
            self._act_dim,
            self._params.hidden_dim,
            self._params.hidden_layers,
            self._params.rmin,
            self._params.rmax,
            self._params.rbins,
        )

        projector = mlp(
            self._params.hidden_dim,
            self._params.projector_hidden_dim,
            self._params.projector_dim,
            hidden_layers=1,
        )
        # todo: check number of hidden layers used in predictor (same as projector??)
        predictor = mlp(
            self._params.projector_dim, self._params.projector_hidden_dim, self._params.projector_dim, hidden_layers=1
        )

        self._muzero = MuZero(
            representation,
            dynamics,
            policy,
            value,
            projector,
            predictor,
            dynamics.distribution,
            value.distribution,
            QNetwork(
                self._act_dim,
                self._params.hidden_dim,
                self._params.hidden_layers,
                self._params.vmin,
                self._params.vmax,
                self._params.vbins,
            ),
        ).to(self._device)

        self._target_muzero = copy.deepcopy(self._muzero)

        self._optimizer = torch.optim.AdamW(
            self._muzero.parameters(),
            lr=1e-3,
            weight_decay=self._params.weight_decay,
        )

        # if True:
        #    self._optimizer = torch.optim.SGD(
        #        self._muzero.parameters(), lr=1e-3, momentum=0.0, weight_decay=self._params.weight_decay
        #    )

    def act(self, observation):
        self._target_muzero.eval()
        obs = observation.float().to(self._device)
        action, policy, q, value = self._target_muzero.act(
            obs,
            self._params.exploration_epsilon,
            self._params.exploration_alpha,
            self._params.mcts_temperature,
            self._params.discount_rate,
            self._params.mcts_depth,
            self._params.mcts_samples,
            self._params.ucb_c1,
            self._params.ucb_c2,
        )
        policy = policy.cpu().numpy()
        value = value.cpu().numpy().item()

        # debug/testing
        if np.random.rand() < 0.1:
            action = np.random.randint(0, self._act_dim)
        # else:
        #    action = torch.argmax(q, dim=1).detach().cpu().item()

        return action, policy, value

    def reanalyze(self, observation):
        return self._target_muzero.reanalyze(
            observation,
            self._params.exploration_epsilon,
            self._params.exploration_alpha,
            self._params.discount_rate,
            self._params.mcts_depth,
            self._params.mcts_samples,
            self._params.ucb_c1,
            self._params.ucb_c2,
        )

    def learn(self, batch):
        self._muzero.train()

        # todo: use schedule
        lr = self._params.learning_rate
        for grp in self._optimizer.param_groups:
            grp["lr"] = lr

        batch_tensors = []
        for i, tensor in enumerate(batch):
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            batch_tensors.append(tensor.to(self._device))
        batch = EpisodeBatch(*batch_tensors)

        # priority = batch.priority[:, 0].view(-1)

        # importance_weight = 1 / (priority + 1e-6) / self._replay_buffer.size()
        importance_weight = batch.importance_weight.view(-1)

        target_value = torch.clamp(batch.target_value, self._params.vmin, self._params.vmax)
        target_reward = torch.clamp(batch.rewards, self._params.rmin, self._params.rmax)

        priority, info = self._muzero.train_step(
            self._optimizer,
            batch.state,
            batch.action,
            target_reward,
            batch.next_state,
            batch.done,
            batch.target_policy,
            target_value,
            importance_weight,
            self._params.max_norm,
            self._params.target_label_smoothing_factor,
            self._params.s_weight,
            self._params.v_weight,
            self._params.discount_rate,
            self._target_muzero,
        )

        online_params = itertools.chain(self._muzero.parameters(), self._muzero.buffers())
        target_params = itertools.chain(self._target_muzero.parameters(), self._target_muzero.buffers())
        for po, pt in zip(online_params, target_params):
            gamma = 0.9
            pt.data = gamma * pt.data + (1 - gamma) * po.data

        for key, val in info.items():
            if isinstance(val, torch.Tensor):
                info[key] = val.detach().cpu().numpy().item()

        # Return loss
        return priority, info

    def _update_target(self):
        pass

    def save(self, f):
        # note: we intentionally do not store the replay buffer and optimizer state
        # as they are not needed for the current training setup
        torch.save(
            {
                "obs_dim": self._obs_dim,
                "act_dim": self._act_dim,
                "training_config": self._params,
                "muzero": self._muzero.state_dict(),
                "target_muzero": self._target_muzero.state_dict(),
            },
            f,
        )

    @staticmethod
    def load(f, device):
        checkpoint = torch.load(f, map_location=device)
        muzero_state_dict = checkpoint.pop("muzero")
        target_muzero_state_dict = checkpoint.pop("target_muzero")
        agent = MuZeroAgent(device=device, **checkpoint)
        agent._muzero.load_state_dict(muzero_state_dict)
        agent._target_muzero.load_state_dict(target_muzero_state_dict)
        return agent


class UniformPolicy(torch.nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.act_dim = act_dim

    def forward(self, x):
        bsz = x.shape[0]
        p = torch.ones((bsz, self.act_dim), dtype=x.dtype, device=x.device)
        p /= torch.sum(p, dim=1, keepdim=True)
        return p
