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

# pylint: skip-file
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
        self.total_num_turns = 5
        self.mode = kwargs["mode"]

        super().__init__(env_spec=self.create_env_spec(**kwargs), num_players=num_players, framestack=framestack)

    def create_env(self, **_kwargs):
        self._env = SimpleDrivingEnv()

    def create_env_spec(self, **_kwargs):
        env_name = "SimpleDriving-v0"
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
        if self.mode == "train":
            self.switch_turn()
            if not self._turn == 0:
                self.goal = 12 * np.ones((2,))
                self.spawn_position = np.random.uniform(-10, 10, (3,))
                self.spawn_orientation = np.random.uniform(-1, 1, (4,))
                self.spawn_position[2] = 0.5
                agent = "alice"
            else:
                agent = "bob"
        elif self.mode == "test":
            self.goal = np.random.uniform(-12, 12, (2,))
            self.spawn_position = np.random.uniform(-10, 10, (3,))
            self.spawn_orientation = np.random.uniform(-1, 1, (4,))
            self.spawn_position[2] = 0.5
            agent = "bob"

        observation = self._env.reset(self.goal, self.spawn_position, self.spawn_orientation, agent)
        return GymObservation(
            observation=np.concatenate(
                (observation["car_qpos"], np.ndarray.flatten(observation["segmentation"].astype("int32")), self.goal)
            ),
            rewards=[0.0],
            current_player=self._turn,
            legal_moves_as_int=[int(self.agent_done)],
            done=self.trial_done,
            info={},
        )

    def step(self, action=None):

        if self.agent_done:
            self.agent_done = False
            gym_observation = self.reset()

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
            if agent == "alice":
                self.goal = observation["car_qpos"][:2] + np.random.uniform(
                    0.6,
                    0.8,
                    [
                        2,
                    ],
                )  # additional noise to accelerate learning
                observation["car_qpos"][:2] = self.goal
            if agent == "bob" and self.current_turn == self.total_num_turns:
                self.trial_done = True
        return GymObservation(
            observation=np.concatenate(
                (observation["car_qpos"], np.ndarray.flatten(observation["segmentation"].astype("int32")), self.goal)
            ),
            current_player=self._turn,
            legal_moves_as_int=[int(self.agent_done)],
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
