import sys
# inserting pybullet-driving-env to the path
# sys.path.insert(1, '/home/ck/pybullet-driving-env')

import gym
# import pybullet_driving_env
from cogment_verse_environment.pybullet_driving_env.envs.simple_driving_env import SimpleDrivingEnv
from cogment_verse_environment.base import BaseEnv, GymObservation
from cogment_verse_environment.env_spec import EnvSpec
import numpy as np


class DrivingEnv(BaseEnv):
    """
        Class for loading pybullet-driving-env
    """

    def __init__(self, *, num_players=2, framestack=1, spawn=[10, 10], **kwargs):
        assert num_players > 1
        self.create_env()

        # for asymmetric selfplay, self._turn is 0 whenever its Bob's turn to play and [1,num_players] is for each of the alice agents
        self._turn = 0
        self._prev_turn = 1
        self.num_players = num_players
        self._env.reset([10] * 2, [10] * 3, [0] * 4)
        self.agent_done = False
        self.trial_done = False
        self.current_turn = 0
        self.total_num_turns = 4

        super().__init__(
            env_spec=self.create_env_spec(**kwargs), num_players=num_players, framestack=framestack
        )

    def create_env(self, **_kwargs):
        self._env = SimpleDrivingEnv()

    def create_env_spec(self, **_kwargs):
        env_name = 'SimpleDriving-v0'
        obs_spaces = self._env.observation_space.spaces
        act_dim = [self._env.action_space.shape]
        act_shape = [self._env.action_space.shape]

        return EnvSpec(
            env_name=env_name,
            obs_dim=[obs_spaces[space].shape for space in obs_spaces],
            act_dim=act_dim,
            act_shape=act_shape,
        )

    def reset(self):
        self.switch_turn()
        if not self._turn == 0:
            self.goal = 12 * np.ones((2,))
            self.spawn_position = np.random.uniform(-10, 10, (3,))
            self.spawn_orientation = np.random.uniform(-1, 1, (4,))
            self.spawn_position[2] = 0.5
            agent = "alice"
        else:
            agent = "bob"

        observation = self._env.reset(self.goal, self.spawn_position, self.spawn_orientation, agent)
        return GymObservation(
            observation=observation['car_qpos'],
            rewards=[0.0],
            current_player=self._turn,
            legal_moves_as_int=[],
            done=False,
            info={},
        )

    def step(self, action=None):

        if self.agent_done:
            gym_observation = self.reset()
            self.agent_done = False

            if self._turn == 0:
                self.current_turn += 1

            return gym_observation

        if not self._turn == 0:
            step_multiplier = 1
            agent = "alice"
        else:
            step_multiplier = 1.5
            agent = "bob"

        observation, reward, self.agent_done, info = self._env.step(action, step_multiplier, agent)

        if self.agent_done:
            self.goal = observation['car_qpos'][:2]
            if agent == "bob" and self.current_turn == self.total_num_turns:
                self.trial_done = True

        return GymObservation(
            observation=observation['car_qpos'],
            current_player=self._turn,
            legal_moves_as_int=[],
            rewards=[reward],
            done=self.trial_done,
            info=info,
        )

    def switch_turn(self):
        if not self._turn == 0:
            # if last player was alice, run bob now
            self._turn = 0
        else:
            # if last player was bob, run one of the alice
            self._turn = (self._prev_turn - 2) % (self.num_players - 1) + 1
            self._prev_turn = (self._prev_turn - 2) % (self.num_players - 1) + 1

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def close(self):
        self._env.close()
