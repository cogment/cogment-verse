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

from cogment_verse_torch_agents.selfplay_td3.replaybuffer import Memory
from cogment_verse_torch_agents.selfplay_td3.model import ActorNetwork, CriticNetwork
import numpy as np
# import tensorflow_probability as tfp
import pickle as pkl
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pylint: disable=C0103
# pylint: disable=W0613
# pylint: disable=W0221
# pylint: disable=W0212
# pylint: disable=W0622
class SelfPlayTD3:
    def __init__(self, model_params=None, **params):

        self._params = params
        self._params['name'] = self._params['id'].split("_")[-1]
        # self._model = PolicyNetwork(self._params["obs_dim"], self._params["act_dim"])
        # self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._params["lr"])
        # self._replay_buffer = Memory(
        #     self._params["obs_dim"], self._params["act_dim"], self._params["max_replay_buffer_size"]
        # )

        # if model_params is not None:
        #     self._model.set_weights(model_params)
        # self._model.trainable = True

        # rl replay buffer
        # if bob: bc replay buffer

        self._actor_network = ActorNetwork(**self._params)
        self._critic_network = CriticNetwork(**self._params)

        self._replay_buffer = Memory(**self._params)

    def act(self, state, goal, grid):

        # if alice: filter observation
        if self._params['name'] == "bob":
            state = np.concatenate([state, goal])
        grid = np.reshape(grid, self._params["grid_shape"])

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        grid = torch.transpose(torch.FloatTensor(grid).to(device).unsqueeze_(0), 1, 3)
        return self._actor_network(state, grid).cpu().data.numpy().flatten() + np.random.normal(0, self._params["SIGMA"], (1, 2))[0]


        # action self._model.model(observation)
        # return action

        # policy = self._model.model(tf.expand_dims(observation, axis=0), training=False)
        # dist = tfp.distributions.Categorical(probs=policy, dtype=tf.float32)
        # action = dist.sample()
        # return int(action.numpy()[0])

        # return np.random.sample(self._params["act_dim"])

    # def get_discounted_rewards(self, rewards):
    #     discounted_rewards = []
    #     sum_rewards = 0
    #
    #     for r in rewards[::-1]:
    #         sum_rewards = r + self._params["gamma"] * sum_rewards
    #         discounted_rewards.append(sum_rewards)
    #
    #     discounted_rewards = np.array(discounted_rewards[::-1])
    #     discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
    #         discounted_rewards.std() + np.finfo(np.float32).eps.item()
    #     )
    #     return discounted_rewards
    #
    # @staticmethod
    # def __loss(prob, actions, Q):
    #     dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    #     log_prob = dist.log_prob(actions)
    #     loss = -log_prob * Q
    #     return tf.reduce_mean(loss)
    #
    def learn(self):
        pass
        # random bacth from rl replaybuffer
        # if bob: random batch from bc replaybuffer

        # critic loss
        # actor_loss
        # if bob: bc_actor_loss

        # grads, learn
        # return some stats

    #     batch = self._replay_buffer.sample()
    #     Q = self.get_discounted_rewards(batch["rewards"])
    #     with tf.GradientTape() as tape:
    #         prob = self._model.model(batch["observations"], training=True)
    #         loss = self.__loss(prob, batch["actions"], Q)
    #     gradients = tape.gradient(loss, self._model.trainable_variables)
    #
    #     self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
    #     self._replay_buffer.reset_replay_buffer()
    #     return {"loss": loss.numpy(), "rewards_mean": batch["rewards"].mean()}
    #
    def consume_training_sample(self, samples):
        """
        Consume a training sample, e.g. store in an internal replay buffer
        """
        self._replay_buffer.add(samples)

    def save(self, f):
        # pkl.dump({"model_params": self._actor_network}, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(self._params['name'], f, pkl.HIGHEST_PROTOCOL)
        return self._params

    @staticmethod
    def load(f, **params):
        # agent_params = pkl.load(f)
        # agent = SelfPlayTD3(model_params=agent_params["model_params"], **params)
        agent = SelfPlayTD3(**params)
        return agent
