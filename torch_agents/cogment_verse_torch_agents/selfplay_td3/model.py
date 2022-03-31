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

import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: skip-file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, **params):
        """Initialize a ActorNetwork.
        Params
        ======
            name (str): Agent name
            obs_dim1 (int): number of features in agent's gpos
            obs_dim2 (int): number of features in agent's goals
            obs_dim3 (int): number of channels in agent's grid

            number_actions (int): number of actions of the agent
        """
        super(ActorNetwork, self).__init__()
        self.agent = params["name"]

        if self.agent == "bob":
            self.state_dim = params["obs_dim1"] + params["obs_dim2"]
        else:
            self.state_dim = params["obs_dim1"]

        self.grid_channels = params["grid_shape"][2]

        self.action_dim = params["act_dim"]
        self.action_scale = torch.FloatTensor(params["action_scale"]).to(device)
        self.action_bias = torch.FloatTensor(params["action_bias"]).to(device)
        self.max_action = params["max_action"]

        self.cnnl1 = nn.Sequential(
            nn.Conv2d(self.grid_channels, 5, (4, 4)),
            nn.LeakyReLU(),
            nn.Conv2d(5, 10, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(10, 20, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(20, 20, (4, 4), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(20, 20, (3, 3)),
            nn.LeakyReLU(),
        )

        self.cnnfc1 = nn.Linear(320, 100)

        self.l1 = nn.Linear(self.state_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(200, 100)
        self.l4 = nn.Linear(100, self.action_dim)

    def forward(self, state, grid):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        grid = self.cnnl1(grid)
        x = grid.view(-1, grid.size(1) * grid.size(2) * grid.size(3))
        x = F.relu(self.cnnfc1(x))

        a = torch.cat((a, x), 1)
        a = F.relu(self.l3(a))
        a = self.l4(a)

        return self.action_scale * torch.tanh(a) + self.action_bias


class CriticNetwork(nn.Module):
    """Initialize a CriticNetwork.
    Params
    ======
        name (str): Agent name
        obs_dim1 (int): number of features in agent's gpos
        obs_dim2 (int): number of features in agent's goals
        obs_dim3 (int): number of channels in agent's grid

        number_actions (int): number of actions of the agent
    """

    def __init__(self, **params):
        super(CriticNetwork, self).__init__()

        self.agent = params["name"]

        if self.agent == "bob":
            self.state_dim = params["obs_dim1"] + params["obs_dim2"]
        else:
            self.state_dim = params["obs_dim1"]

        self.grid_channels = params["grid_shape"][2]
        self.action_dim = params["act_dim"]

        # Q1 architecture
        self.cnnl1 = nn.Sequential(
            nn.Conv2d(self.grid_channels, 5, (4, 4)),
            nn.LeakyReLU(),
            nn.Conv2d(5, 10, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(10, 20, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(20, 20, (4, 4), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(20, 20, (3, 3)),
            nn.LeakyReLU(),
        )

        self.cnnfc1 = nn.Linear(320, 100)

        self.l1 = nn.Linear(self.state_dim + self.action_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(200, 100)
        self.l4 = nn.Linear(100, 1)

        # Q2 architecture
        self.cnnl2 = nn.Sequential(
            nn.Conv2d(self.grid_channels, 5, (4, 4)),
            nn.LeakyReLU(),
            nn.Conv2d(5, 10, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(10, 20, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(20, 20, (4, 4), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(20, 20, (3, 3)),
            nn.LeakyReLU(),
        )

        self.cnnfc2 = nn.Linear(320, 100)

        self.l5 = nn.Linear(self.state_dim + self.action_dim, 100)
        self.l6 = nn.Linear(100, 100)
        self.l7 = nn.Linear(200, 100)
        self.l8 = nn.Linear(100, 1)

    def forward(self, state, action, grid):
        sa = torch.cat([state, action], 1)
        # print(sa.shape)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        grid1 = self.cnnl1(grid)
        x1 = grid1.view(-1, grid1.size(1) * grid1.size(2) * grid1.size(3))
        x1 = F.relu(self.cnnfc1(x1))
        q1 = torch.cat((q1, x1), 1)
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        grid2 = self.cnnl2(grid)
        x2 = grid2.view(-1, grid2.size(1) * grid2.size(2) * grid2.size(3))
        x2 = F.relu(self.cnnfc2(x2))
        q2 = torch.cat((q2, x2), 1)
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2

    def Q1(self, state, action, grid):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        grid1 = self.cnnl1(grid)
        x1 = grid1.view(-1, grid1.size(1) * grid1.size(2) * grid1.size(3))
        x1 = F.relu(self.cnnfc1(x1))
        q1 = torch.cat((q1, x1), 1)
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1
