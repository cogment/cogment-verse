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
from cogment_verse_torch_agents.selfplay_td3.replaybuffer import Memory
from cogment_verse_torch_agents.selfplay_td3.model import ActorNetwork, CriticNetwork
import numpy as np
import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pylint: disable=C0103
# pylint: disable=W0613
# pylint: disable=W0221
# pylint: disable=W0212
# pylint: disable=W0622
class SelfPlayTD3:
    def __init__(self, model_params=None, **params):

        self._params = params
        self._params["name"] = self._params["id"].split("_")[-1]

        self._actor_network = ActorNetwork(**self._params)
        self._critic_network = CriticNetwork(**self._params)

        self._actor_target_network = copy.deepcopy(self._actor_network)
        self._critic_target_network = copy.deepcopy(self._critic_network)

        self._actor_optimizer = torch.optim.Adam(self._actor_network.parameters(), lr=self._params["learning_rate"])
        self._critic_optimizer = torch.optim.Adam(self._critic_network.parameters(), lr=self._params["learning_rate"])

        self._replay_buffer = Memory(**self._params)
        self.total_it = 0

    def act(self, state, goal, grid):

        # if alice: filter observation
        if self._params["name"] == "bob":
            state = np.concatenate([state, goal])
        grid = np.reshape(grid, self._params["grid_shape"])

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        grid = torch.transpose(torch.FloatTensor(grid).to(device).unsqueeze_(0), 1, 3)
        # TBD refactor noise
        action = (
            self._actor_network(state, grid).cpu().data.numpy().flatten()
            + np.random.normal(0, self._params["SIGMA"], (1, 2))[0]
        )
        return action

    def prepare_data(self, data):
        if self._params["name"] == "bob":
            state = np.concatenate([data["state"], data["goal"]], axis=1)
            next_state = np.concatenate([data["next_state"], data["next_goal"]], axis=1)
        else:
            state = data["state"]
            next_state = data["next_state"]

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)

        grid = np.reshape(
            data["grid"],
            [
                data["grid"].shape[0],
                self._params["grid_shape"][0],
                self._params["grid_shape"][1],
                self._params["grid_shape"][2],
            ],
        )
        grid = torch.transpose(torch.from_numpy(grid).to(device).float(), 1, 3)

        next_grid = np.reshape(
            data["next_grid"],
            [
                data["next_grid"].shape[0],
                self._params["grid_shape"][0],
                self._params["grid_shape"][1],
                self._params["grid_shape"][2],
            ],
        )
        next_grid = torch.transpose(torch.from_numpy(next_grid).to(device).float(), 1, 3)

        action = torch.FloatTensor(data["action"]).to(device)
        reward = torch.FloatTensor(data["reward"]).to(device)
        done = torch.FloatTensor(data["player_done"]).to(device)

        return state, grid, action, reward, next_state, next_grid, done

    def train(self, alice):
        self.total_it += 1
        training_batch = self._replay_buffer.sample()
        state, grid, action, reward, next_state, next_grid, done = self.prepare_data(training_batch)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self._params["policy_noise"]).clamp(
                -self._params["noise_clip"], self._params["noise_clip"]
            )

            next_action = (self._actor_target_network(next_state, next_grid) + noise).clamp(
                -self._params["max_action"], self._params["max_action"]
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self._critic_target_network(next_state, next_action, next_grid)
            target_Q = torch.min(target_Q1, target_Q2)
            # print(target_Q.shape, not_done.shape)
            target_Q = reward.reshape(target_Q.shape) + (1.0 - done) * self._params["discount_factor"] * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self._critic_network(state, action, grid)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self._params["policy_freq"] == 0:

            # Compute actor loss
            actions = self._actor_network(state, grid)
            actor_loss = -self._critic_network.Q1(state, actions, grid).mean()
            if self._params["name"] == "bob":
                # print(state.shape)
                alice_actions = alice._actor_network(state[:, :7], grid[:, :2, :, :])
                actor_loss = actor_loss + self._params["beta"] * F.mse_loss(actions, alice_actions).mean()

            # Optimize the actor
            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self._critic_network.parameters(), self._critic_target_network.parameters()):
                target_param.data.copy_(
                    self._params["tau"] * param.data + (1 - self._params["tau"]) * target_param.data
                )

            for param, target_param in zip(self._actor_network.parameters(), self._actor_target_network.parameters()):
                target_param.data.copy_(
                    self._params["tau"] * param.data + (1 - self._params["tau"]) * target_param.data
                )

            return actor_loss.item(), critic_loss.item() / self._params["batch_size"]
        return 0, critic_loss.item() / self._params["batch_size"]

    def learn(self, alice=None):
        mean_actor_loss, mean_critic_loss = 0, 0
        if self._replay_buffer.get_size() >= self._params["min_buffer_size"]:
            actor_loss, critic_loss = 0, 0
            for _ in range(self._params["num_training_steps"]):
                actor_loss_, critic_loss_ = self.train(alice)
                actor_loss += actor_loss_
                critic_loss += critic_loss_

            mean_actor_loss = actor_loss / self._params["num_training_steps"]
            mean_critic_loss = critic_loss / self._params["num_training_steps"]

        return {"mean_actor_loss": mean_actor_loss, "mean_critic_loss": mean_critic_loss}

    def consume_samples(self, samples):
        """
        Consume a training sample, e.g. store in an internal replay buffer
        """
        self._replay_buffer.add(samples)

    def save(self, f):
        torch.save((self._actor_network, self._critic_network), f)
        return {}

    @staticmethod
    def load(f, **params):
        (actor_network, critic_network) = torch.load(f)
        agent = SelfPlayTD3(**params)

        agent._critic_network = critic_network
        agent._critic_target_network = copy.deepcopy(agent._critic_network)

        agent._actor_network = actor_network
        agent._actor_target_network = copy.deepcopy(agent._actor_network)

        return agent
