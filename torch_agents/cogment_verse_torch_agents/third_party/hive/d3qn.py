import copy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


from .utils.utils import get_optimizer_fn
from .utils.schedule import PeriodicSchedule, get_schedule
from .agent import Agent



class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        torch.manual_seed(0)
        layer_size = 32

        self.value = nn.Sequential(
            nn.Linear(state_size, layer_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(layer_size, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(state_size, layer_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(layer_size, action_size)
        )


    def forward(self, state):
        values = self.value(state)
        advantages = self.advantage(state)
        q_values = values + (advantages - advantages.mean())

        return q_values



class D3QNAgent(Agent):
    """An agent implementing the D3QN algorithm. Uses an epsilon greedy
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
        seed=42,
        device="cpu",
        max_replay_buffer_size=50000,
    ):
        """
        Args:
            qnet: A network that outputs the q-values of the different actions
                for an input observation.
            obs_dim: The dimension of the observations.
            act_dim: The number of actions available to the agent.
            optimizer_fn: A function that takes in a list of parameters to optimize
                and returns the optimizer.
            id: ID used to create the timescale in the logger for the agent.
            replay_buffer: The replay buffer that the agent will push observations
                to and sample from during learning.
            discount_rate (float): A number between 0 and 1 specifying how much
                future rewards are discounted by the agent.
            grad_clip (float): Gradients will be clipped to between
                [-grad_clip, gradclip]
            target_net_soft_update (bool): Whether the target net parameters are
                replaced by the qnet parameters completely or using a weighted
                average of the target net parameters and the qnet parameters.
            target_net_update_fraction (float): The weight given to the target
                net parameters in a soft update.
            target_net_update_schedule: Schedule determining how frequently the
                target net is updated.
            epsilon_schedule: Schedule determining the value of epsilon through
                the course of training.
            learn_schedule: Schedule determining when the learning process actually
                starts.
            seed: Seed for numpy random number generator.
            batch_size (int): The size of the batch sampled from the replay buffer
                during learning.
            device: Device on which all computations should be run.
        """
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
        self._params["discount_rate"] = discount_rate
        self._params["grad_clip"] = grad_clip
        self._params["target_net_soft_update"] = target_net_soft_update
        self._params["target_net_update_fraction"] = target_net_update_fraction

        self._device = torch.device(device)

        self._qnet = QNetwork(obs_dim, act_dim)
        self._target_qnet = QNetwork(obs_dim, act_dim)


        optimizer_fn = get_optimizer_fn(optimizer_fn)
        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adam
        self._optimizer = optimizer_fn(self._qnet.parameters())

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
    def act(self, observation, legal_moves_as_int, update_schedule=True):
        self.eval()

        observation = torch.tensor(observation).to(self._device).float()
        legal_moves_as_int = torch.tensor(legal_moves_as_int).to(self._device).float()

        observation = torch.tensor(np.expand_dims(observation, axis=0)).to(self._device).float()

        # if not self._params["distributional"]:
        epsilon = self.get_epsilon_schedule(update_schedule)

        if self._rng.random() < epsilon:
            legal_moves = torch.nonzero(legal_moves_as_int == 0).view(-1).cpu().numpy()
            action = self._rng.choice(legal_moves)
        else:
            with torch.no_grad():
                a = self._qnet(observation).cpu() # , legal_moves=legal_moves_as_int
            action = torch.argmax(a).numpy()

        return action

    def learn(self, batch, update_schedule=True):
        info = {}
        self.train()

        info["lr"] = self._lr_schedule.update()
        for grp in self._optimizer.param_groups:
            grp["lr"] = info["lr"]

        # do not modify batch in-place
        batch = {key: torch.tensor(value).to(self._device) for key, value in batch.items()}

        # Compute predicted Q values
        self._optimizer.zero_grad()
        pred_qvals = self._qnet(batch["observations"]) # , legal_moves=batch["legal_moves_as_int"]
        actions = batch["actions"].long()

        pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

        # Compute 1-step Q targets
        next_qvals = self._target_qnet(batch["next_observations"])
        next_qvals, _ = torch.max(next_qvals, dim=1)

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
