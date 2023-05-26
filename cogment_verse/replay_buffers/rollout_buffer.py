import torch


class RolloutBuffer:
    """Rollout buffer for PPO"""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple,
        action_shape: tuple,
        observation_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.observation_dtype = observation_dtype
        self.action_dtype = action_dtype
        self.reward_dtype = reward_dtype

        self.observations = torch.zeros((self.capacity, *self.observation_shape), dtype=self.observation_dtype)
        self.actions = torch.zeros((self.capacity, *self.action_shape), dtype=self.action_dtype)
        self.rewards = torch.zeros((self.capacity,), dtype=self.reward_dtype)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32)

        self._ptr = 0
        self.num_total = 0

    def add(self, observation: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor) -> None:
        """Add samples to rollout buffer"""
        if self.num_total < self.capacity:
            self.observations[self._ptr] = observation
            self.actions[self._ptr] = action
            self.rewards[self._ptr] = reward
            self.dones[self._ptr] = done
            self._ptr = (self._ptr + 1) % self.capacity
            self.num_total += 1

    def reset(self) -> None:
        """Reset the rollout"""
        self.observations = torch.zeros((self.capacity, *self.observation_shape), dtype=self.observation_dtype)
        self.actions = torch.zeros((self.capacity, *self.action_shape), dtype=self.action_dtype)
        self.rewards = torch.zeros((self.capacity,), dtype=self.reward_dtype)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32)
        self._ptr = 0
        self.num_total = 0
