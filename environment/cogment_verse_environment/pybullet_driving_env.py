import sys
# inserting pybullet-driving-env to the path
sys.path.insert(1, '/home/ck/pybullet-driving-env')

import gym
import pybullet_driving_env
from cogment_verse_environment.base import BaseEnv, GymObservation
from cogment_verse_environment.env_spec import EnvSpec


class DrivingEnv(BaseEnv):
    """
        Class for loading pybullet-driving-env
    """
    def __init__(self, *, num_players=1, framestack=1, **kwargs):
        self.create_env()
        super().__init__(
            env_spec=self.create_env_spec(**kwargs), num_players=num_players, framestack=framestack
        )

    def create_env(self, **_kwargs):
        self._env = gym.make('SimpleDriving-v0')

    def create_env_spec(self, **_kwargs):
        env_name = 'SimpleDriving-v0'
        obs_spaces = self._env.observation_space.spaces
        act_dim = [self._env.action_space.shape]        
        act_shape = [self._env.action_space.shape]        
        
        return EnvSpec(
            env_name=env_name,
            obs_dim=[space.shape for space in obs_spaces],
            act_dim=act_dim,
            act_shape=act_shape,
        )
    
    def reset(self, goal, spawn_position, spawn_orientation, agent = "alice"):
        observation = self._env.reset(goal, spawn_position, spawn_orientation, agent)
        return GymObservation(
            observation=observation,
            rewards=[0.0],
            current_player=self._turn,
            legal_moves_as_int=[],
            done=False,
            info={},
        )

    def step(self, action = None, step_multiplier = 1, agent = "alice"):
        observation, reward, done, info = self._env.step(action, step_multiplier, agent)
        return GymObservation(
            observation=observation,
            current_player=self._turn,
            legal_moves_as_int=[],
            rewards=[reward],
            done=done,
            info=info,
        )
    
    def seed(self, seed = None):
        self._env.seed(seed=seed)
    
    def close(self):
        self._env.close()