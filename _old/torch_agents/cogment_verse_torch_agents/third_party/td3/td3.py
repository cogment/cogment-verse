# This code has been modified from its original version found at
# https://github.com/sfujim/TD3

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from cogment_verse_torch_agents.third_party.hive.agent import Agent
from cogment_verse_torch_agents.third_party.hive.utils.schedule import PeriodicSchedule
from cogment_verse_torch_agents.third_party.td3.td3_mlp import ActorMLP, CriticMLP


class TD3Agent(Agent):
    """An agent implementing the DQN algorithm. Uses an epsilon greedy
    exploration policy
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        high_action=[1, 1],
        low_action=[-1, -1],
        start_timesteps=2000,
        expl_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        optimizer_fn=None,
        id=0,
        replay_buffer=None,
        discount_rate=0.99,
        target_net_update_fraction=0.005,
        target_net_update_schedule=None,
        epsilon_schedule=None,
        learn_schedule=None,
        lr_schedule=None,
        seed=42,
        device="cpu",
        logger=None,
        log_frequency=100,
        max_replay_buffer_size=50000,
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            id=id,
            seed=seed,
            learn_schedule=learn_schedule,
            epsilon_schedule=epsilon_schedule,
            lr_schedule=lr_schedule,
            max_replay_buffer_size=max_replay_buffer_size,
        )

        # TODO fix this high variable
        self._actor = ActorMLP(obs_dim, act_dim).to(device)
        self._actor_target = copy.deepcopy(self._actor)
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=3e-4)

        self._critic = CriticMLP(obs_dim, act_dim).to(device)
        self._critic_target = copy.deepcopy(self._critic)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=3e-4)

        self._params["obs_dim"] = obs_dim
        self._params["act_dim"] = act_dim
        self._params["max_action"] = torch.tensor(high_action).to(device)
        self._params["min_action"] = torch.tensor(low_action).to(device)

        self._params["policy_noise"] = policy_noise
        self._params["noise_clip"] = noise_clip
        self._params["policy_freq"] = policy_freq

        self._params["discount_rate"] = discount_rate
        self._params["tau"] = target_net_update_fraction

        self._device = torch.device(device)  # pylint: disable=no-member
        self._loss_fn = torch.nn.SmoothL1Loss()

        self._target_net_update_schedule = PeriodicSchedule(off_value=False, on_value=True, period=100)

        self._state = {"episode_start": True}
        self._training = True
        self._params["total_it"] = 0

        self._params["start_timesteps"] = start_timesteps
        self._params["expl_noise"] = expl_noise

        self._id = id

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._actor.train()
        self._actor_target.train()
        self._critic.train()
        self._critic_target.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._actor.eval()
        self._actor_target.eval()
        self._critic.eval()
        self._critic_target.eval()

    @torch.no_grad()
    def act(self, observation, legal_moves_as_int=None, update_schedule=True):
        self.eval()

        epsilon = self.get_epsilon_schedule(update_schedule)

        observation = torch.tensor(np.expand_dims(observation, axis=0)).to(self._device).float()
        if self._params["total_it"] < self._params["start_timesteps"] or self._rng.random() < epsilon:
            uniform_action = torch.rand(self._params["act_dim"])
            span = self._params["max_action"] - self._params["min_action"]
            action = uniform_action * span + self._params["min_action"]
        else:
            span = self._params["max_action"] - self._params["min_action"]
            center = span * 0.5
            loc = self._params["min_action"] + center
            scale = center * self._params["expl_noise"]
            action_actor = (
                self._actor(observation, self._params["min_action"], self._params["max_action"]).cpu().numpy()
            )
            noise = np.random.normal(loc, scale, size=self._params["act_dim"])

            action = action_actor + noise
            action = action.clip(self._params["min_action"], self._params["max_action"])

        return action

    def learn(self, batch, update_schedule=True):
        info = {}
        self._params["total_it"] += 1

        self.train()

        # Learn rate schedule for optimizer
        info["lr"] = self._lr_schedule.update()
        for grp in self._actor_optimizer.param_groups + self._critic_optimizer.param_groups:
            grp["lr"] = info["lr"]

        batch = {key: torch.tensor(val).to(self._device) for key, val in batch.items()}

        with torch.no_grad():
            next_action = self._actor_target(
                batch["next_observations"], self._params["min_action"], self._params["max_action"]
            )
            action_shape = next_action.shape
            batch["actions"] = batch["actions"].reshape(action_shape)
            noise = (torch.randn_like(next_action) * self._params["policy_noise"]).clamp(
                -self._params["noise_clip"], self._params["noise_clip"]
            )  # pylint: disable=no-member

            next_action = torch.max(
                torch.min(next_action + noise, self._params["max_action"]), self._params["min_action"]
            )

            target_Q1, target_Q2 = self._critic_target(batch["next_observations"], next_action)
            target_Q = torch.min(target_Q1, target_Q2)  # pylint: disable=no-member
            target_Q = batch["rewards"] + self._params["discount_rate"] * (1 - batch["done"]) * target_Q.squeeze(dim=1)
            target_Q = torch.unsqueeze(target_Q, dim=1)  # pylint: disable=no-member

        current_Q1, current_Q2 = self._critic(batch["observations"], batch["actions"])

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        info["critic_loss"] = critic_loss.item()

        if self._params["total_it"] % self._params["policy_freq"] == 0:
            actor_loss = -self._critic.Q1(
                batch["observations"],
                self._actor(batch["observations"], self._params["min_action"], self._params["max_action"]),
            ).mean()

            self._actor_optimizer.zero_grad()
            actor_loss.backward()
            self._actor_optimizer.step()
            info["actor_loss"] = actor_loss.item()

            self._update_target()

        if update_schedule:
            self.get_epsilon_schedule(update_schedule)

        info["exploration"] = int(self._params["total_it"] < self._params["start_timesteps"])
        return info

    def _update_target(self):
        for param, target_param in zip(self._critic.parameters(), self._critic_target.parameters()):
            target_param.data.copy_(self._params["tau"] * param.data + (1 - self._params["tau"]) * target_param.data)

        for param, target_param in zip(self._actor.parameters(), self._actor_target.parameters()):
            target_param.data.copy_(self._params["tau"] * param.data + (1 - self._params["tau"]) * target_param.data)

    def save(self, f):
        torch.save(
            {
                "id": self._id,
                "params": self._params,
                "state": self._state,
                "learn_schedule": self._learn_schedule,
                "epsilon_schedule": self._epsilon_schedule,
                "target_net_update_schedule": self._target_net_update_schedule,
                "rng": self._rng,
                "actor": self._actor.state_dict(),
                "actor_target": self._actor_target.state_dict(),
                "actor_optimizer": self._actor_optimizer.state_dict(),
                "critic": self._critic.state_dict(),
                "critic_target": self._critic_target.state_dict(),
                "critic_optimizer": self._critic_optimizer.state_dict(),
                "lr_schedule": self._lr_schedule,
            },
            f,
        )

    def load(self, f):
        super().load(f)
        checkpoint = torch.load(f, map_location=self._device)

        self._id = checkpoint["id"]
        self._params = checkpoint["params"]
        self._state = checkpoint["state"]

        self._params["min_action"] = self._params["min_action"].to(self._device)
        self._params["max_action"] = self._params["max_action"].to(self._device)

        self._learn_schedule = checkpoint["learn_schedule"]
        self._epsilon_schedule = checkpoint["epsilon_schedule"]
        self._target_net_update_schedule = checkpoint["target_net_update_schedule"]

        self._rng = checkpoint["rng"]

        self._actor.load_state_dict(checkpoint["actor"])
        self._actor_target.load_state_dict(checkpoint["actor_target"])
        self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        self._critic.load_state_dict(checkpoint["critic"])
        self._critic_target.load_state_dict(checkpoint["critic_target"])
        self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        self._lr_schedule = checkpoint["lr_schedule"]
