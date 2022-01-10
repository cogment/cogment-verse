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

from cogment_verse_torch_agents.muzero.networks import (
    MuZero,
    lin_bn_act,
    mlp,
    RepresentationNetwork,
    PolicyNetwork,
    ValueNetwork,
    DynamicsNetwork,
)
from cogment_verse_torch_agents.muzero.replay_buffer import EpisodeBatch

# pylint: disable=arguments-differ
# pylint: disable=invalid-name

log = logging.getLogger(__name__)


class MuZeroAgent:
    """
    MuZero implementation
    """

    def __init__(self, *, obs_dim, act_dim, device, training_config: MuZeroTrainingConfig):
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self.params = training_config
        self._device = torch.device(device)
        self._make_networks()

    def set_device(self, device):
        self._device = torch.device(device)
        self.muzero = self.muzero.to(self._device)
        self.target_muzero = self.target_muzero.to(self._device)

    def _make_networks(self):
        stem = lin_bn_act(self._obs_dim, self.params.hidden_dim, bn=True, act=torch.nn.ReLU())
        representation = RepresentationNetwork(stem, self.params.hidden_dim, self.params.hidden_layers)
        policy = PolicyNetwork(self.params.hidden_dim, self.params.hidden_layers, self._act_dim)

        value = ValueNetwork(
            self.params.hidden_dim,
            self.params.hidden_layers,
            self.params.vmin,
            self.params.vmax,
            self.params.vbins,
        )

        dynamics = DynamicsNetwork(
            self._act_dim,
            self.params.hidden_dim,
            self.params.hidden_layers,
            self.params.rmin,
            self.params.rmax,
            self.params.rbins,
        )

        projector = mlp(
            self.params.hidden_dim,
            self.params.projector_hidden_dim,
            self.params.projector_dim,
            hidden_layers=1,
        )
        # todo: check number of hidden layers used in predictor (same as projector??)
        predictor = mlp(
            self.params.projector_dim, self.params.projector_hidden_dim, self.params.projector_dim, hidden_layers=1
        )

        self.muzero = MuZero(
            representation,
            dynamics,
            policy,
            value,
            projector,
            predictor,
            dynamics.distribution,
            value.distribution,
        ).to(self._device)

        self.target_muzero = copy.deepcopy(self.muzero)

        self._optimizer = torch.optim.AdamW(
            self.muzero.parameters(),
            lr=1e-3,
            weight_decay=self.params.weight_decay,
        )

    def act(self, observation):
        self.target_muzero.eval()
        obs = observation.clone().detach().float().to(self._device)
        action, policy, _q, value = self.target_muzero.act(
            obs,
            self.params.exploration_epsilon,
            self.params.exploration_alpha,
            self.params.mcts_temperature,
            self.params.discount_rate,
            self.params.mcts_depth,
            self.params.mcts_samples,
            self.params.ucb_c1,
            self.params.ucb_c2,
        )
        policy = policy.cpu().numpy()
        value = value.cpu().numpy().item()

        return action, policy, value

    def reanalyze(self, observation):
        return self.target_muzero.reanalyze(
            observation.clone().to(self._device),
            self.params.exploration_epsilon,
            self.params.exploration_alpha,
            self.params.discount_rate,
            self.params.mcts_depth,
            self.params.mcts_samples,
            self.params.ucb_c1,
            self.params.ucb_c2,
            self.params.mcts_temperature,
        )

    def learn(self, batch):
        self.muzero.train()

        # todo: use schedule
        lr = self.params.learning_rate
        for grp in self._optimizer.param_groups:
            grp["lr"] = lr

        batch_tensors = []
        for tensor in batch:
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            batch_tensors.append(tensor.to(self._device))
        batch = EpisodeBatch(*batch_tensors)

        priority, info = self.muzero.train_step(
            self._optimizer,
            batch.state,
            batch.action,
            batch.target_reward_probs,
            batch.target_reward,
            batch.next_state,
            batch.done,
            batch.target_policy,
            batch.target_value_probs,
            batch.target_value,
            batch.importance_weight,
            self.params.max_norm,
            self.params.s_weight,
            self.params.v_weight,
            self.params.discount_rate,
            self.target_muzero,
        )

        online_params = itertools.chain(self.muzero.parameters(), self.muzero.buffers())
        target_params = itertools.chain(self.target_muzero.parameters(), self.target_muzero.buffers())
        for po, pt in zip(online_params, target_params):
            gamma = 0.9
            pt.data = gamma * pt.data + (1 - gamma) * po.data

        for key, val in info.items():
            if isinstance(val, torch.Tensor):
                info[key] = val.detach().cpu().numpy().item()

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
                "training_config": self.params,
                "muzero": self.muzero.state_dict(),
                "target_muzero": self.target_muzero.state_dict(),
            },
            f,
        )

    @staticmethod
    def load(f, device):
        checkpoint = torch.load(f, map_location=device)
        muzero_state_dict = checkpoint.pop("muzero")
        target_muzero_state_dict = checkpoint.pop("target_muzero")
        agent = MuZeroAgent(device=device, **checkpoint)
        agent.muzero.load_state_dict(muzero_state_dict)
        agent.target_muzero.load_state_dict(target_muzero_state_dict)
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
