import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from cogment_verse_torch_agents.third_party.hive.agent import Agent
from cogment_verse_torch_agents.third_party.td3.td3_mlp import ActorMLP, CriticMLP


class DDPGAgent(Agent):
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
        target_net_update_fraction=1e-3,
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

        self._params["obs_dim"] = obs_dim
        self._params["act_dim"] = act_dim

        self._params["device"] = device

        self._params["policy_noise"] = policy_noise
        self._params["noise_clip"] = noise_clip
        self._params["policy_freq"] = policy_freq

        self._params["discount_rate"] = discount_rate
        self._params["tau"] = target_net_update_fraction

        self._device = torch.device(device)  # pylint: disable=no-member
        self._params["min_action"] = torch.tensor(low_action).to(self._device)
        self._params["max_action"] = torch.tensor(high_action).to(self._device)

        LR_ACTOR = 1e-4
        self._actor_local = ActorMLP(obs_dim, act_dim).to(device)
        self._actor_target = ActorMLP(obs_dim, act_dim).to(self._device)
        self._actor_optimizer = optim.Adam(self._actor_local.parameters(), lr=LR_ACTOR)

        LR_CRITIC = 3e-4
        WEIGHT_DECAY = 0.0001
        self.critic_local = CriticMLP(obs_dim, act_dim).to(self._device)
        self._critic_target = CriticMLP(obs_dim, act_dim).to(self._device)
        self._critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self._params["total_it"] = 0
        self._params["start_timesteps"] = start_timesteps
        self._params["expl_noise"] = expl_noise

    def train(self):
        """Changes the agent to training mode."""
        super().train()
        self._actor_local.train()
        self._actor_target.train()
        self.critic_local.train()
        self._critic_target.train()

    def eval(self):
        """Changes the agent to evaluation mode."""
        super().eval()
        self._actor_local.eval()
        self._actor_target.eval()
        self.critic_local.eval()
        self._critic_target.eval()

    def act(self, state, legal_moves_as_int=None, update_schedule=True):
        state = torch.from_numpy(np.array(state, copy=True)).float().to(self._device)

        assert len(state.shape) <= 2
        if len(state.shape) == 2:
            assert state.shape[0] == 1
            assert state.shape[1] == self._params["obs_dim"]
        else:
            state = state.unsqueeze(0)

        self.eval()
        epsilon = self.get_epsilon_schedule(update_schedule)
        if self._params["total_it"] < self._params["start_timesteps"] or self._rng.random() < epsilon:
            uniform_action = torch.rand(self._params["act_dim"])
            span = self._params["max_action"] - self._params["min_action"]
            action = uniform_action * span + self._params["min_action"]
        else:
            with torch.no_grad():
                action = (
                    self._actor_local(state, self._params["min_action"], self._params["max_action"]).cpu().data.numpy()
                )

            span = self._params["max_action"] - self._params["min_action"]
            center = span * 0.5
            loc = self._params["min_action"] + center
            scale = center * self._params["expl_noise"]
            noise = np.random.normal(loc, scale, size=self._params["act_dim"])

            action += noise
            action = np.clip(action, self._params["min_action"], self._params["max_action"])

        return action

    def learn(self, experiences, update_schedule=True):
        self._params["total_it"] += 1
        batch = {key: torch.tensor(val).to(self._device) for key, val in experiences.items()}

        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        dones = batch["done"]

        actions_next = self._actor_target(next_states, self._params["min_action"], self._params["max_action"])

        action_shape = actions_next.shape
        actions = actions.reshape(action_shape)

        Q_targets_next = self._critic_target.Q1(next_states, actions_next)
        Q_targets = rewards + (self._params["discount_rate"] * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local.Q1(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        actions_pred = self._actor_local(states, self._params["min_action"], self._params["max_action"])
        actor_loss = -self.critic_local.Q1(states, actions_pred).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        self.soft_update(self.critic_local, self._critic_target)
        self.soft_update(self._actor_local, self._actor_target)

        if update_schedule:
            self.get_epsilon_schedule(update_schedule)

        info = {}
        info["critic_loss"] = critic_loss.item()
        info["actor_loss"] = actor_loss.item()
        info["exploration"] = int(self._params["total_it"] < self._params["start_timesteps"])

        return info

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self._params["tau"] * local_param.data + (1.0 - self._params["tau"]) * target_param.data
            )

    def save(self, f):
        torch.save(
            {
                "id": self._id,
                "params": self._params,
                "learn_schedule": self._learn_schedule,
                "epsilon_schedule": self._epsilon_schedule,
                "rng": self._rng,
                "actor": self._actor_local.state_dict(),
                "actor_target": self._actor_target.state_dict(),
                "actor_optimizer": self._actor_optimizer.state_dict(),
                "critic": self.critic_local.state_dict(),
                "critic_target": self._critic_target.state_dict(),
                "critic_optimizer": self._critic_optimizer.state_dict(),
                "lr_schedule": self._lr_schedule,
            },
            f,
        )

    def load(self, f):
        super().load(f)
        device_name = self._params["device"]
        checkpoint = torch.load(f, map_location=self._device)

        self._id = checkpoint["id"]
        self._params = checkpoint["params"]
        self._params["device"] = device_name
        self._learn_schedule = checkpoint["learn_schedule"]
        self._epsilon_schedule = checkpoint["epsilon_schedule"]
        self._rng = checkpoint["rng"]

        for key in ["min_action", "max_action"]:
            self._params[key] = checkpoint["params"][key].to(self._device)

        self._actor_local.load_state_dict(checkpoint["actor"])
        self._actor_target.load_state_dict(checkpoint["actor_target"])
        self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])

        self.critic_local.load_state_dict(checkpoint["critic"])
        self._critic_target.load_state_dict(checkpoint["critic_target"])
        self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        self._lr_schedule = checkpoint["lr_schedule"]
