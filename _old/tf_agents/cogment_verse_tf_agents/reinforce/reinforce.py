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

import pickle as pkl

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from cogment_verse_tf_agents.reinforce.model import PolicyNetwork
from cogment_verse_tf_agents.reinforce.replaybuffer import Memory


# pylint: disable=C0103
# pylint: disable=W0613
# pylint: disable=W0221
# pylint: disable=W0212
# pylint: disable=W0622
class ReinforceAgent:
    def __init__(self, model_params=None, **params):

        self._params = params
        self._model = PolicyNetwork(self._params["obs_dim"], self._params["act_dim"])
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._params["lr"])
        self._replay_buffer = Memory(
            self._params["obs_dim"], self._params["act_dim"], self._params["max_replay_buffer_size"]
        )

        if model_params is not None:
            self._model.set_weights(model_params)
        self._model.trainable = True

    def act(self, observation):

        policy = self._model.model(tf.expand_dims(observation, axis=0), training=False)
        dist = tfp.distributions.Categorical(probs=policy, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def get_discounted_rewards(self, rewards):
        discounted_rewards = []
        sum_rewards = 0

        for r in rewards[::-1]:
            sum_rewards = r + self._params["gamma"] * sum_rewards
            discounted_rewards.append(sum_rewards)

        discounted_rewards = np.array(discounted_rewards[::-1])
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + np.finfo(np.float32).eps.item()
        )
        return discounted_rewards

    @staticmethod
    def __loss(prob, actions, Q):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(actions)
        loss = -log_prob * Q
        return tf.reduce_mean(loss)

    def learn(self):

        batch = self._replay_buffer.sample()
        Q = self.get_discounted_rewards(batch["rewards"])
        with tf.GradientTape() as tape:
            prob = self._model.model(batch["observations"], training=True)
            loss = self.__loss(prob, batch["actions"], Q)
        gradients = tape.gradient(loss, self._model.trainable_variables)

        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        self._replay_buffer.reset_replay_buffer()
        return {"loss": loss.numpy(), "rewards_mean": batch["rewards"].mean()}

    def consume_training_sample(self, sample):
        """
        Consume a training sample, e.g. store in an internal replay buffer
        """
        self._replay_buffer.add(sample)

    def save(self, f):
        pkl.dump({"model_params": self._model.get_weights()}, f, pkl.HIGHEST_PROTOCOL)
        return self._params

    @staticmethod
    def load(f, **params):
        agent_params = pkl.load(f)
        agent = ReinforceAgent(model_params=agent_params["model_params"], **params)
        return agent
