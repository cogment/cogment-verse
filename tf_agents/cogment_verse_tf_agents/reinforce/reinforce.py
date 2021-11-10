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

from cogment_verse_tf_agents.reinforce.model import PolicyNetwork
from cogment_verse_tf_agents.third_party.hive.replay_buffer import CircularReplayBuffer
from cogment_verse_tf_agents.third_party.hive.utils.schedule import ConstantSchedule

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# pylint: disable=C0103
# pylint: disable=W0613
# pylint: disable=W0221
# pylint: disable=W0212
# pylint: disable=W0622
class ReinforceAgent:
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        id,
        obs_dim,
        act_dim,
        epsilon=0.01,
        gamma=0.95,
        lr=3e-4,
        max_replay_buffer_size=50000,
        seed=42,
        epsilon_schedule=None,
        lr_schedule=None,
    ):
        self.id = id
        self._params = {}
        self._params["obs_dim"] = obs_dim
        self._params["act_dim"] = act_dim
        self._params["seed"] = seed
        self._params["max_replay_buffer_size"] = max_replay_buffer_size
        self._params["gamma"] = gamma

        self._epsilon_schedule = epsilon_schedule
        if epsilon_schedule is None:
            self._epsilon_schedule = ConstantSchedule(epsilon)

        self._lr_schedule = lr_schedule
        if lr_schedule is None:
            self._lr_schedule = ConstantSchedule(lr)

        self._model = PolicyNetwork(self._params["obs_dim"], self._params["act_dim"])
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._lr_schedule.get_value())

        self._replay_buffer = CircularReplayBuffer(
            seed=self._params["seed"], size=self._params["max_replay_buffer_size"]
        )

    def regularize_dist(self, prob):
        prob = prob + self._epsilon_schedule.get_value()
        return prob / tf.reduce_sum(prob, axis=1)[:, tf.newaxis]

    def act(self, observation, legal_moves_as_int=None, update_schedule=True):

        policy = self._model.model(tf.expand_dims(observation, axis=0), training=False)
        policy = self.regularize_dist(policy)
        dist = tfp.distributions.Categorical(probs=policy, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def get_discounted_rewards(self, rewards):
        discounted_rewards = []
        sum_rewards = 0

        for r in rewards[::-1]:
            sum_rewards = r + self._params["gamma"] * sum_rewards
            discounted_rewards.append(sum_rewards)

        return np.array(discounted_rewards[::-1])

    @staticmethod
    def __loss(prob, actions, Q):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(actions)
        loss = -log_prob * Q
        return tf.reduce_mean(loss)

    def learn(self, batch, update_schedule=True):

        Q = self.get_discounted_rewards(batch["rewards"])
        with tf.GradientTape() as tape:
            prob = self._model.model(batch["observations"], training=True)
            prob = self.regularize_dist(prob)
            loss = self.__loss(prob, batch["actions"], Q)
        gradients = tape.gradient(loss, self._model.trainable_variables)

        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        return {"loss": loss}

    def consume_training_sample(self, sample):
        """
        Consume a training sample, e.g. store in an internal replay buffer
        """
        self._replay_buffer.add(sample)

    def sample_training_batch(self, batch_size=32):
        """
        sample last trial SARSD

        Args:
            batch_size (int): .
        """
        indices = range(self._replay_buffer._n)
        rval = {}
        for key, _ in self._replay_buffer._data.items():
            rval[key] = np.asarray([self._replay_buffer._data[key][idx] for idx in indices], dtype="float32")

        return rval

    def reset_replay_buffer(self):
        """
        Resets buffer
        """
        self._replay_buffer._write_index = -1
        self._replay_buffer._n = 0

    def init_params(self, model_parms):
        self._model.set_weights(model_parms)
        self._model.trainable = True

    def get_replay_buffer_size(self):
        """
        Return the size of the internal replay buffer
        """
        return self._replay_buffer.size()
