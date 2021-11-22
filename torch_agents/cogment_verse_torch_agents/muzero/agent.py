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

log = logging.getLogger(__name__)

from .networks import MuZero, reward_transform, reward_tansform_inverse, Distributional, DynamicsAdapter, resnet, mlp
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

    def _make_networks(self):
        value_distribution = Distributional(
            self._params.vmin,
            self._params.vmax,
            self._params.hidden_dim,
            self._params.vbins,
            reward_transform,
            reward_tansform_inverse,
        )

        reward_distribution = Distributional(
            self._params.rmin,
            self._params.rmax,
            self._params.hidden_dim,
            self._params.rbins,
            reward_transform,
            reward_tansform_inverse,
        )

        representation = resnet(
            self._obs_dim,
            self._params.hidden_dim,
            self._params.representation_dim,
            self._params.hidden_layers,
            # final_act=torch.nn.BatchNorm1d(self._params.num_latent"]),  # normalize for input to subsequent networks
        )

        dynamics = DynamicsAdapter(
            resnet(
                self._params.representation_dim + self._act_dim,
                self._params.hidden_dim,
                self._params.hidden_dim,
                self._params.hidden_layers - 1,
                final_act=torch.nn.LeakyReLU(),
            ),
            self._act_dim,
            self._params.hidden_dim,
            self._params.representation_dim,
            reward_dist=reward_distribution,
        )
        policy = resnet(
            self._params.representation_dim,
            self._params.hidden_dim,
            self._act_dim,
            self._params.hidden_layers,
            final_act=torch.nn.Softmax(dim=1),
        )
        value = resnet(
            self._params.representation_dim,
            self._params.hidden_dim,
            self._params.hidden_dim,
            self._params.hidden_layers - 1,
            final_act=value_distribution,
        )
        projector = mlp(
            self._params.representation_dim,
            self._params.projector_hidden_dim,
            self._params.projector_dim,
            hidden_layers=self._params.projector_hidden_layers,
        )
        # todo: check number of hidden layers used in predictor (same as projector??)
        predictor = mlp(
            self._params.projector_dim, self._params.projector_hidden_dim, self._params.projector_dim, hidden_layers=1
        )

        self._muzero = MuZero(
            representation, dynamics, policy, value, projector, predictor, reward_distribution, value_distribution
        ).to(self._device)

        self._optimizer = torch.optim.AdamW(
            self._muzero.parameters(),
            lr=1e-3,
            weight_decay=self._params.weight_decay,
        )

    def act(self, observation):
        self._muzero.eval()
        obs = observation.float().to(self._device)
        action, policy, value = self._muzero.act(
            obs, self._params.exploration_epsilon, self._params.exploration_alpha, self._params.mcts_temperature
        )
        policy = policy.cpu().numpy()
        value = value.cpu().numpy().item()
        return action, policy, value

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
            batch.target_policy,
            target_value,
            importance_weight,
            self._params.max_norm,
        )

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
            },
            f,
        )

    @staticmethod
    def load(f, device):
        checkpoint = torch.load(f, map_location=device)
        muzero_state_dict = checkpoint.pop("muzero")
        agent = MuZeroAgent(device=device, **checkpoint)
        agent._muzero.load_state_dict(muzero_state_dict)
        return agent
