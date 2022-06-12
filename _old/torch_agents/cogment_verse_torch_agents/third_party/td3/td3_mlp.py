# This code has been modified from its original version found at
# https://github.com/sfujim/TD3

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ActorMLP(nn.Module):
    """Simple Actor MLP function approximator for TD3"""

    def __init__(self, in_dim, out_dim, hidden_units=256, num_hidden_layers=1):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(np.prod(in_dim), hidden_units), nn.ReLU())
        self.hidden_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU()) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_units, out_dim)

    def forward(self, x, low, high):
        x = torch.flatten(x, start_dim=1)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return 0.5 * (low + high) + 0.5 * (high - low) * torch.tanh(self.output_layer(x))


class CriticMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CriticMLP, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(in_dim + out_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(in_dim + out_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
