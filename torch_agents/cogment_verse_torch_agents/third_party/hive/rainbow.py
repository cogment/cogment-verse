import copy

import numpy as np
import torch

from .agent import Agent
from .dqn import legal_moves_adapter
from .mlp import ComplexMLP, DistributionalMLP
from .utils.schedule import PeriodicSchedule, get_schedule
from .utils.utils import get_optimizer_fn


class RainbowDQNAgent(Agent):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        *,
        obs_dim,
        act_dim,
        v_min=0,
        v_max=200,
        atoms=51,
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
        double=True,
        dueling=True,
        noisy=True,
        distributional=True,
        max_replay_buffer_size=50000,
    ):
        """
        Args:
            qnet: A network that outputs the q-values of the different actions
                for an input observation.
            obs_dim: The dimension of the observations.
            act_dim: The number of actions available to the agent.
            v_min: minimum possible value of the value function
            v_max: maximum possible value of the value function
            atoms: number of atoms in the distributional DQN context
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
            double: whether or not to use the double feature (from double DQN)
            distributional: whether or not to use the distributional feature (from distributional DQN)
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
        self._params["double"] = double
        self._params["dueling"] = dueling
        self._params["noisy"] = noisy
        self._params["distributional"] = distributional

        self._params["discount_rate"] = discount_rate
        self._params["grad_clip"] = grad_clip
        self._params["target_net_soft_update"] = target_net_soft_update
        self._params["target_net_update_fraction"] = target_net_update_fraction

        self._device = torch.device(device)

        # qnet = {}
        # qnet['kwargs'] = {}

        if self._params["distributional"]:
            self._params["atoms"] = atoms
            self._params["v_min"] = v_min
            self._params["v_max"] = v_max
            self._supports = torch.linspace(self._params["v_min"], self._params["v_max"], self._params["atoms"]).to(
                self._device
            )
            # qnet["kwargs"]["supports"] = self._supports
            self._delta = float(self._params["v_max"] - self._params["v_min"]) / (self._params["atoms"] - 1)
            self._nsteps = 1

        if self._params["distributional"]:
            self._qnet = legal_moves_adapter(DistributionalMLP)(
                self._params["obs_dim"],
                self._params["act_dim"],
                self._supports,
                hidden_units=256,
                num_hidden_layers=2,
                noisy=False,
                dueling=True,
                sigma_init=0.5,
                atoms=atoms,
            ).to(self._device)
        else:
            self._qnet = legal_moves_adapter(ComplexMLP)(
                self._params["obs_dim"],
                self._params["act_dim"],
                hidden_units=256,
                num_hidden_layers=2,
                noisy=self._params["noisy"],
                dueling=self._params["dueling"],
                sigma_init=0.4,
                atoms=1,
            ).to(self._device)

        self._target_qnet = copy.deepcopy(self._qnet).requires_grad_(False)

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

    def projection_distribution(self, batch):
        batch_obs = batch["observations"]
        batch_next_obs = batch["next_observations"]
        batch_reward = batch["rewards"].reshape(-1, 1).to(self._device)
        batch_not_done = 1 - batch["done"].reshape(-1, 1).to(self._device)

        with torch.no_grad():
            next_action = self._target_qnet(batch_next_obs).argmax(1)
            next_dist = self._target_qnet.dist(batch_next_obs)
            next_dist = next_dist[range(batch["observations"].shape[0]), next_action]

            t_z = batch_reward + batch_not_done * self._params["discount_rate"] * self._supports
            t_z = t_z.clamp(min=self._params["v_min"], max=self._params["v_max"])
            b = (t_z - self._params["v_min"]) / self._delta
            l = b.floor().long()
            u = b.ceil().long()

            l[(u > 0) * (l == u)] -= 1
            u[(l < (self._params["atoms"] - 1)) * (l == u)] += 1

            offset = (
                torch.linspace(0, (batch_obs.shape[0] - 1) * self._params["atoms"], batch_obs.shape[0])
                .long()
                .unsqueeze(1)
                .expand(batch_obs.shape[0], self._params["atoms"])
                .to(self._device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self._device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

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

        observation = torch.tensor(observation).to(self._device).float()
        formatted_legal_moves = torch.tensor(formatted_legal_moves).to(self._device).float()

        observation = torch.tensor(np.expand_dims(observation, axis=0)).to(self._device).float()

        # if not self._params["distributional"]:
        epsilon = self.get_epsilon_schedule(update_schedule)

        if self._rng.random() < epsilon:
            legal_moves = torch.nonzero(formatted_legal_moves == 0).view(-1).cpu().numpy()
            action = self._rng.choice(legal_moves)
        else:
            a = self._qnet(observation, legal_moves=formatted_legal_moves).cpu()
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
        pred_qvals = self._qnet(batch["observations"], legal_moves=batch["legal_moves_as_int"])
        actions = batch["actions"].long()

        if self._params["distributional"]:
            # todo: need legal moves??
            current_dist = self._qnet.dist(batch["observations"])
            log_p = torch.log(current_dist[range(batch["observations"].shape[0]), actions])
            target_prob = self.projection_distribution(batch)

            loss = -(target_prob * log_p).sum(1)
            loss = loss.mean()

        else:
            pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

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
        checkpoint = torch.load(f)
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
