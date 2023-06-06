# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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

from data_pb2 import ObserverAction, PlayerAction, TeacherAction  # pylint: disable=import-error
from gymnasium.spaces import utils

from ..constants import OBSERVER_ACTOR_CLASS, PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS
from .ndarray_serialization import deserialize_ndarray, serialize_ndarray

ACTOR_CLASS_ACTION_CLASS = {
    TEACHER_ACTOR_CLASS: TeacherAction,
    PLAYER_ACTOR_CLASS: PlayerAction,
    OBSERVER_ACTOR_CLASS: ObserverAction,
}

# pylint: disable=attribute-defined-outside-init
class Action:
    """
    Cogment Verse actor action

    Properties:
        flat_value:
            The action value, as a flat numpy array.
        value:
            The action value, as a numpy array.
    """

    def __init__(self, gym_space, pb_action=None, value=None):
        """
        Action constructor.
        Shouldn't be called directly, prefer the factory function of ActionSpace.
        """
        self._gym_space = gym_space

        if pb_action is not None:
            assert value is None
            self._pb_action = pb_action
            return

        self._value = value

    def _compute_flat_value(self):
        if hasattr(self, "_value"):
            value = self._value
            if value is None:
                return None
            return utils.flatten(self._gym_space, self._value)

        if not self._pb_action.HasField("value"):
            # This happens whenever value is None
            return None

        return deserialize_ndarray(self._pb_action.value)

    @property
    def flat_value(self):
        if not hasattr(self, "_flat_value"):
            self._flat_value = self._compute_flat_value()
        return self._flat_value

    def _compute_value(self):
        flat_value = self.flat_value
        if flat_value is None:
            return None
        return utils.unflatten(self._gym_space, flat_value)

    @property
    def value(self):
        if not hasattr(self, "_value"):
            self._value = self._compute_value()
        return self._value


class ActionSpace:
    """
    Cogment Verse action space

    Properties:
        gym_space:
            Wrapped Gym space for the action values (cf. https://www.gymlibrary.dev/api/spaces/)
        actor_class:
            Class of the actor for which this space will serialize Action probobug messages
        seed:
            Random seed used when generating random actions
    """

    def __init__(self, gym_space, actor_class=PLAYER_ACTOR_CLASS, seed=None):
        self.gym_space = gym_space
        self._action_class = ACTOR_CLASS_ACTION_CLASS[actor_class]

        if seed:
            self.gym_space.seed(int(seed))

    def create(self, value=None):
        """
        Create an Action
        """
        return Action(self.gym_space, value=value)

    def sample(self, mask=None):
        """
        Generate a random Action
        """
        return Action(self.gym_space, value=self.gym_space.sample(mask=mask))

    def serialize(
        self,
        action,
    ):
        """
        Serialize an Action to an Action protobuf message
        """
        if action.value is None:
            return self._action_class()

        serialized_value = serialize_ndarray(action.flat_value)
        return self._action_class(value=serialized_value)

    def deserialize(self, pb_action):
        """
        Deserialize an Action from an Action protobuf message
        """
        return Action(self.gym_space, pb_action=pb_action)
