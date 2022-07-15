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

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# pylint: disable=R1725
class PolicyNetwork(Model):
    def __init__(self, number_features, number_actions):
        """Initialize a Model object.
        Params
        ======
            number_features (int): number of features in player's observation
            number_actions (int): number of actions of the player
        """
        super(PolicyNetwork, self).__init__()
        self.number_features = number_features
        self.number_actions = number_actions

        self.model = tf.keras.Sequential(
            [
                layers.Input(shape=(self.number_features)),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(self.number_actions, activation="softmax"),
            ]
        )

    def call(self, x):
        action = self.model(x)
        return action
