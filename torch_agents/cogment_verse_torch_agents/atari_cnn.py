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

import copy

import numpy as np
import torch
from cogment_verse_torch_agents.third_party.hive.agent import Agent
from cogment_verse_torch_agents.third_party.hive.dqn import legal_moves_adapter
from cogment_verse_torch_agents.third_party.hive.utils.schedule import PeriodicSchedule, get_schedule
from cogment_verse_torch_agents.third_party.hive.utils.utils import get_optimizer_fn
from torch import nn

# TODO address those issues properly
# pylint: disable=redefined-builtin,arguments-differ,arguments-renamed


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kernel_h, kernel_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride,) * 2
    padding_h, padding_w = padding if isinstance(padding, tuple) else (padding,) * 2
    h = (h + (2 * padding_h) - (dilation * (kernel_h - 1)) - 1) // stride_h + 1
    w = (w + (2 * padding_w) - (dilation * (kernel_w - 1)) - 1) // stride_w + 1
    return h, w


class SimpleConvModel(nn.Module):
    """
    Simple convolutional network approximator for Q-Learning.
    """

    def __init__(self, in_dim, out_dim, channels, kernel_sizes, strides, paddings, mlp_layers):
        """
        Args:
            in_dim (tuple): The tuple of observations dimension (channels, width, height)
            out_dim (int): The action dimension
            channels (list): The size of output channel for each convolutional layer
            kernel_sizes (list): The kernel size for each convolutional layer
            strides (list): The stride used for each convolutional layer
            paddings (list): The size of the padding used for each convolutional layer
            mlp_layers (list): The size of neurons for each mlp layer after the convolutional layers
        """

        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(strides)
        assert len(channels) == len(paddings)

        super().__init__()

        c, h, w = in_dim

        # Default Convolutional Layers
        channels = [c] + channels

        conv_layers = [
            torch.nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=ks,
                stride=s,
                padding=p,
            )
            for (in_c, out_c, ks, s, p) in zip(channels[:-1], channels[1:], kernel_sizes, strides, paddings)
        ]
        conv_seq = []
        for conv_layer in conv_layers:
            conv_seq.extend([conv_layer, torch.nn.ReLU()])
        self.conv = torch.nn.Sequential(*conv_seq)

        # Default MLP Layers
        conv_out_size = self.conv_out_size(h, w)
        head_units = [conv_out_size] + mlp_layers + [out_dim]
        head_layers = [torch.nn.Linear(i, o) for i, o in zip(head_units[:-1], head_units[1:])]
        head_seq = []
        for head_layer in head_layers:
            head_seq.extend([head_layer, torch.nn.ReLU()])
        self.head = torch.nn.Sequential(*head_seq)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        x = x.type(torch.float) / 255.0
        conv_out = self.conv(x)
        return self.head(conv_out.view(batch_size, -1))

    def conv_out_size(self, h, w, c=None):
        """
        Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model.
        """
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size, child.stride, child.padding)
            except AttributeError:
                pass
            try:
                c = child.out_channels
            except AttributeError:
                pass
        return h * w * c


# pylint disavle
class NatureAtariDQNModel(Agent):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        *,
        obs_dim,
        act_dim,
        optimizer_fn=None,
        id=0,
        discount_rate=0.99,
        grad_clip=None,
        target_net_soft_update=False,
        target_net_update_fraction=0.05,
        target_net_update_schedule=None,
        epsilon_schedule=None,
        learn_schedule=None,
        lr_schedule=None,
        double=True,
        seed=42,
        device="cpu",
        framestack=4,
        screensize=84,
        weight_decay=1e-4,
        max_replay_buffer_size=50000,
    ):
        super().__init__(
            id=id,
            seed=seed,
            obs_dim=obs_dim,
            act_dim=act_dim,
            learn_schedule=learn_schedule,
            epsilon_schedule=epsilon_schedule,
            lr_schedule=lr_schedule,
            max_replay_buffer_size=max_replay_buffer_size,
        )
        self._params["double"] = double
        self._params["discount_rate"] = discount_rate
        self._params["grad_clip"] = grad_clip
        self._params["target_net_soft_update"] = target_net_soft_update
        self._params["target_net_update_fraction"] = target_net_update_fraction
        self._params["weight_decay"] = weight_decay

        self._params["device"] = device
        self._device = torch.device(device)

        self._qnet = legal_moves_adapter(SimpleConvModel)(
            in_dim=[framestack, screensize, screensize],
            out_dim=act_dim,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            paddings=[0, 1, 1],
            mlp_layers=[512],
        ).to(device)
        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

        optimizer_fn = get_optimizer_fn(optimizer_fn)
        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        self._optimizer = optimizer_fn(self._qnet.parameters(), weight_decay=weight_decay)

        self._loss_fn = torch.nn.SmoothL1Loss()

        self._id = id

        self._target_net_update_schedule = get_schedule(target_net_update_schedule)
        if self._target_net_update_schedule is None:
            self._target_net_update_schedule = PeriodicSchedule(False, True, 10000)

        self._state = {"episode_start": True}
        self._training = True

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._qnet.train()
        self._target_qnet.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._qnet.eval()
        self._target_qnet.eval()

    @torch.no_grad()
    def act(self, observation, formatted_legal_moves, update_schedule=True):
        self.eval()

        formatted_legal_moves = torch.tensor(formatted_legal_moves).to(self._device).float()
        observation = torch.tensor(np.expand_dims(observation, axis=0)).to(self._device).float()

        # if not self._params["distributional"]:
        epsilon = self.get_epsilon_schedule(update_schedule)

        if self._rng.random() < epsilon:
            legal_moves = torch.nonzero(formatted_legal_moves == 0).view(-1).cpu().numpy()
            action = self._rng.choice(legal_moves)
        else:
            action_qs = self._qnet(observation, legal_moves=formatted_legal_moves).cpu()
            action = torch.argmax(action_qs).cpu().numpy()

        return action

    def learn(self, batch, update_schedule=True):
        info = {}
        self.train()

        info["lr"] = self._lr_schedule.update()
        for grp in self._optimizer.param_groups:
            grp["lr"] = info["lr"]

        # do not modify batch in-place
        batch = copy.copy(batch)
        for key, tensor in batch.items():
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            batch[key] = tensor.to(self._device)

        # Compute predicted Q values
        self._optimizer.zero_grad()
        pred_qvals = self._qnet(batch["observations"], legal_moves=batch["legal_moves_as_int"])
        actions = batch["actions"].long()
        pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

        with torch.no_grad():
            # Compute 1-step Q targets
            if self._params["double"]:
                next_action = self._qnet(batch["next_observations"], legal_moves=batch["legal_moves_as_int"])
            else:
                next_action = self._target_qnet(batch["next_observations"], legal_moves=batch["legal_moves_as_int"])

            _, next_action = torch.max(next_action, dim=1)
            next_qvals = self._target_qnet(batch["next_observations"])
            next_qvals = next_qvals[torch.arange(next_qvals.size(0)), next_action]

            q_targets = batch["rewards"] + self._params["discount_rate"] * next_qvals * (1 - batch["done"])

        loss = self._loss_fn(pred_qvals, q_targets)

        if self._training:
            loss.backward()
            if self._params["grad_clip"] is not None:
                torch.nn.utils.clip_grad_value_(self._qnet.parameters(), self._params["grad_clip"])
            self._optimizer.step()

        # Update target network
        if self._training and self._target_net_update_schedule.update():
            self._update_target()

        if update_schedule:
            self.get_epsilon_schedule(update_schedule)

        # Return loss
        info["loss"] = loss.item()
        return info

    def _update_target(self):
        if self._params["target_net_soft_update"]:
            target_params = self._target_qnet.state_dict()
            current_params = self._qnet.state_dict()
            for key in list(target_params.keys()):
                target_params[key] = (1 - self._params["target_net_update_fraction"]) * target_params[
                    key
                ] + self._params["target_net_update_fraction"] * current_params[key]
            self._target_qnet.load_state_dict(target_params)
        else:
            self._target_qnet.load_state_dict(self._qnet.state_dict())

    def save(self, f):
        torch.save(
            {
                "id": self._id,
                "params": self._params,
                "qnet": self._qnet.state_dict(),
                "target_qnet": self._target_qnet.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "learn_schedule": self._learn_schedule,
                "epsilon_schedule": self._epsilon_schedule,
                "target_net_update_schedule": self._target_net_update_schedule,
                "rng": self._rng,
                "lr_schedule": self._lr_schedule,
            },
            f,
        )

    def load(self, f):
        super().load(f)
        checkpoint = torch.load(f, map_location=self._device)
        checkpoint["device"] = self._params["device"]

        self._id = checkpoint["id"]
        self._params = checkpoint["params"]
        self._qnet.load_state_dict(checkpoint["qnet"])
        self._target_qnet.load_state_dict(checkpoint["target_qnet"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._learn_schedule = checkpoint["learn_schedule"]
        self._epsilon_schedule = checkpoint["epsilon_schedule"]
        self._target_net_update_schedule = checkpoint["target_net_update_schedule"]
        self._rng = checkpoint["rng"]
        self._lr_schedule = checkpoint["lr_schedule"]
