import abc

import numpy as np
from cogment_verse_torch_agents.third_party.hive.replay_buffer import CircularReplayBuffer

from .utils.schedule import CosineSchedule, LinearSchedule, SwitchSchedule, get_schedule


class Agent(abc.ABC):
    """Base class for agents. Every implemented agent should be a subclass of this class."""

    def __init__(
        self,
        *,
        obs_dim,
        act_dim,
        max_replay_buffer_size,
        id=0,
        seed=42,
        epsilon_schedule=None,
        learn_schedule=None,
        lr_schedule=None,
    ):
        """Constructor for Agent class.
        Args:
            obs_dim: dimension of observations that agent will see.
            act_dim: Number of actions that the agent needs to chose from.
            id: Identifier for the agent.
        """
        self._id = str(id)
        self._version_number = None
        self._version_hash = None
        self._params = {}
        self._params["obs_dim"] = obs_dim
        self._params["act_dim"] = act_dim
        self._params["seed"] = seed
        self._training = True
        self._params["max_replay_buffer_size"] = max_replay_buffer_size

        self._rng = np.random.default_rng(seed=self._params["seed"])

        self._epsilon_schedule = get_schedule(epsilon_schedule)
        if self._epsilon_schedule is None:
            self._epsilon_schedule = LinearSchedule(1, 0.1, 100000)

        self._learn_schedule = get_schedule(learn_schedule)
        if self._learn_schedule is None:
            self._learn_schedule = SwitchSchedule(False, True, 5000)

        self._lr_schedule = get_schedule(lr_schedule)
        if self._lr_schedule is None:
            self._lr_schedule = CosineSchedule(0.0, 1e-4, 1000)

        self._create_replay_buffer()

    def _create_replay_buffer(self):
        """
        Create the replay buffer. Can be overridden for algorithms that
        require different kinds of replay buffers.
        """
        self._replay_buffer = CircularReplayBuffer(
            seed=self._params["seed"], size=self._params["max_replay_buffer_size"]
        )

    def id(self):
        return self._id

    def version_number(self):
        return self._version_number

    def version_hash(self):
        return self._version_hash

    def set_version_info(self, version_number, version_hash):
        self._version_number = version_number
        self._version_hash = version_hash

    def consume_training_sample(self, sample):
        """
        Consume a training sample, e.g. store in an internal replay buffer
        """
        self._replay_buffer.add(sample)

    def sample_training_batch(self, batch_size):
        """
        Take a sample from the internal replay buffer and return it
        """
        return self._replay_buffer.sample(batch_size)

    def replay_buffer_size(self):
        """
        Return the size of the internal replay buffer
        """
        return self._replay_buffer.size()

    def reset_replay_buffer(self):
        """
        Reset replay buffer. To be implemented by the agent.
        """
        pass

    @abc.abstractmethod
    def act(self, observation, *args, **kwargs):
        """Returns an action for the agent to perform based on the observation"""
        pass

    @abc.abstractmethod
    def learn(self, batch):
        self.train(True)
        pass

    def train(self, mode=True):
        self._training = mode

    def eval(self):
        """Changes the agent to evaluation mode"""
        # self.train(False)
        pass

    @abc.abstractmethod
    def save(self, dname):
        """
        Saves agent checkpointing information to file for future loading.

        Args:
            dname: directory where agent should save all relevant info.
        """
        pass

    @abc.abstractmethod
    def load(self, dname):
        """
        Loads agent information from file.

        Args:
            dname: directory where agent checkpoint info is stored.

        Returns:
            True if successfully loaded agent. False otherwise.
        """
        pass

    def get_epsilon_schedule(self, update_schedule=False):
        if self._training:
            if not self._learn_schedule.update():
                epsilon = 1.0
            else:
                if update_schedule:
                    epsilon = self._epsilon_schedule.update()
                else:
                    epsilon = self._epsilon_schedule.get_value()
        else:
            epsilon = 0

        return epsilon
