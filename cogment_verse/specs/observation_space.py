# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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

from gym.spaces import utils, Dict

from data_pb2 import Observation as PbObservation  # pylint: disable=import-error

from .encode_rendered_frame import encode_rendered_frame
from .ndarray_serialization import serialize_ndarray, deserialize_ndarray

# pylint: disable=attribute-defined-outside-init
class Observation:
    """
    Cogment Verse actor observation

    Properties:
        flat_value:
            The observation value, as a flat numpy array.
        value:
            The observation value, as a numpy array.
        flat_action_mask: optional
            The action mask, as a flat numpy array.
        action_mask: optional
            The action mask.
        rendered_frame: optional
            Environmnent's rendered frame as a numpy array of RGB pixels
        current_player: optional
            Name of the current player. `None` for single player environments.
        overridden_players:
            List of players whose actions where overriden by a teacher actor.
    """

    def __init__(
        self,
        gym_space,
        action_mask_gym_space,
        pb_observation=None,
        value=None,
        action_mask=None,
        rendered_frame=None,
        current_player=None,
        overridden_players=None,
    ):
        """
        Observation constructor.
        Shouldn't be called directly, prefer the factory function of ObservationSpace.
        """

        self._gym_space = gym_space
        self._action_mask_gym_space = action_mask_gym_space

        if pb_observation is not None:
            assert value is None
            assert action_mask is None
            assert rendered_frame is None
            assert current_player is None
            assert overridden_players is None
            self._pb_observation = pb_observation
            return

        self._value = value
        self._action_mask = action_mask
        self._rendered_frame = rendered_frame

        self._pb_observation = PbObservation(
            current_player=current_player,
            overridden_players=overridden_players,
        )

    def _compute_flat_value(self):
        if hasattr(self, "_value"):
            return utils.flatten(self._gym_space, self._value)

        return deserialize_ndarray(self._pb_observation.value)

    @property
    def flat_value(self):
        if not hasattr(self, "_flat_value"):
            self._flat_value = self._compute_flat_value()
        return self._flat_value

    def _compute_value(self):
        return utils.unflatten(self._gym_space, self.flat_value)

    @property
    def value(self):
        if not hasattr(self, "_value"):
            self._value = self._compute_value()
        return self._value

    def _compute_flat_action_mask(self):
        if hasattr(self, "_action_mask"):
            if self._action_mask is None:
                return None
            return utils.flatten(self._action_mask_gym_space, self._action_mask)

        if not self._pb_observation.HasField("action_mask"):
            return None

        return deserialize_ndarray(self._pb_observation.action_mask)

    @property
    def flat_action_mask(self):
        if not hasattr(self, "_flat_action_mask"):
            self._flat_action_mask = self._compute_flat_action_mask()
        return self._flat_action_mask

    def _compute_action_mask(self):
        flat_action_mask = self.flat_action_mask
        if flat_action_mask is None:
            return None
        return utils.unflatten(self._action_mask_gym_space, self.flat_action_mask)

    @property
    def action_mask(self):
        if not hasattr(self, "_action_mask"):
            self._action_mask = self._compute_action_mask()
        return self._action_mask

    @property
    def rendered_frame(self):
        try:
            return self._rendered_frame
        except AttributeError as exc:
            # At the moment there's no use to deserialize the rendered frame on the python side
            raise NotImplementedError from exc

    @property
    def current_player(self):
        if not self._pb_observation.HasField("current_player"):
            return None

        return self._pb_observation.current_player

    @property
    def overridden_players(self):
        overridden_players = self._pb_observation.overridden_players
        if overridden_players is None:
            return []
        return overridden_players


class ObservationSpace:
    """
    Cogment Verse observation space

    Properties:
        gym_space:
            Wrapped Gym space for the observation values (cf. https://www.gymlibrary.dev/api/spaces/)
        action_mask_gym_space: optional
            Wrapped Gym space for the action mask (cf. https://www.gymlibrary.dev/api/spaces/)
        render_width:
            Maximum width for the serialized rendered frame in observations
    """

    def __init__(self, space, render_width=1024):
        """
        ObservationSpace constructor.
        Shouldn't be called directly, prefer the factory function of EnvironmentSpecs.
        """
        if isinstance(space, Dict) and ("action_mask" in space.spaces):
            # Check the observation space defines an action_mask "component" (like petting zoo does)
            assert "observation" in space.spaces
            assert len(space.spaces) == 2

            self.gym_space = space.spaces["observation"]
            self.action_mask_gym_space = space.spaces["action_mask"]
        else:
            # "Standard" observation space, no action_mask
            self.gym_space = space
            self.action_mask_gym_space = None

        # Other configuration
        self.render_width = render_width

    def create(self, value=None, action_mask=None, rendered_frame=None, current_player=None, overridden_players=None):
        """
        Create an Observation
        """
        return Observation(
            self.gym_space,
            self.action_mask_gym_space,
            value=value,
            action_mask=action_mask,
            rendered_frame=rendered_frame,
            current_player=current_player,
            overridden_players=overridden_players,
        )

    def serialize(
        self,
        observation,
    ):
        """
        Serialize an Observation to an Observation protobuf message
        """
        flat_value = utils.flatten(self.gym_space, observation.value)
        serialized_value = serialize_ndarray(flat_value)

        serialized_action_mask = None
        if self.action_mask_gym_space is not None:
            flat_action_mask = utils.flatten(self.action_mask_gym_space, observation.action_mask)
            serialized_action_mask = serialize_ndarray(flat_action_mask)

        serialized_rendered_frame = None
        if observation.rendered_frame is not None:
            serialized_rendered_frame = encode_rendered_frame(observation.rendered_frame, self.render_width)

        return PbObservation(
            value=serialized_value,
            action_mask=serialized_action_mask,
            rendered_frame=serialized_rendered_frame,
            overridden_players=observation.overridden_players,
            current_player=observation.current_player,
        )

    def deserialize(self, pb_observation):
        """
        Deserialize an Observation from an Observation protobuf message
        """
        return Observation(self.gym_space, self.action_mask_gym_space, pb_observation=pb_observation)
