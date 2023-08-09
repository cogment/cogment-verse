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

import os

import yaml
from data_pb2 import ActorSpecs as PbActorSpecs  # pylint: disable=import-error
from google.protobuf.json_format import MessageToDict, ParseDict

from ..constants import PLAYER_ACTOR_CLASS, ActorSpecType
from .action_space import ActionSpace
from .ndarray_serialization import SerializationFormat
from .observation_space import ObservationSpace
from .spaces_serialization import deserialize_gym_space, serialize_gym_space


class ActorSpecs:
    """
    Representation of the specification of an environment within cogment verse for a specific observation space.
    Properties:
        spec_type:
            The name identifier for all actors with this observation space and action space.
        web_components_file:
            File with environment controls for web app.
    """

    def __init__(self, actor_specs_pb):
        """
        ActorSpecs constructor.
        Shouldn't be called directly, prefer the factory function such as ActorSpecs.deserialize or ActorSpecs.create.
        """
        self._pb = actor_specs_pb

    def __str__(self):
        return f"ActorSpecs: actor_spec = {self.actor_specs}, web_components_file = {self.web_components_file}"

    @property
    def spec_type(self):
        if hasattr(self._pb, "specs"):
            return ActorSpecType.from_config(self._pb.actor_spec)
        else:
            return ActorSpecType.DEFAULT  # Backwards compatibility with single actor specs setup

    @property
    def web_components_file(self):
        return self._pb.web_components_file

    def get_observation_space(self, render_width=1024):
        """
        Build an instance of the observation space for this actor
        Parameters:
            render_width: optional
                maximum width for the serialized rendered frame in observation
        """
        return ObservationSpace(deserialize_gym_space(self._pb.observation_space), render_width)

    def get_action_space(self, actor_class=PLAYER_ACTOR_CLASS, seed=None):
        """
        Build an instance of the action space for this actor
        Parameters:
            actor_class: optional
                the class of the actor for which we want to retrieve the action space.
                this parameters is mostly useful when serializing actions.
            seed: optional
                the seed used when generating random actions
        """
        return ActionSpace(deserialize_gym_space(self._pb.action_space), actor_class, seed)

    @classmethod
    def create(
        cls,
        observation_space,
        action_space,
        web_components_file,
        spec_type=ActorSpecType.DEFAULT,
        serialization_format=SerializationFormat.STRUCTURED,
    ):
        """
        Factory function building an ActorSpecs.
        """
        return cls.deserialize(
            PbActorSpecs(
                spec_type=spec_type.value,
                observation_space=serialize_gym_space(observation_space, serialization_format),
                action_space=serialize_gym_space(action_space, serialization_format),
                web_components_file=web_components_file,
            )
        )

    def serialize(self):
        """
        Serialize to a EnvironmentSpecs protobuf message
        """
        return self._pb

    @classmethod
    def deserialize(cls, actor_specs_pb):
        """
        Factory function building an ActorSpecs instance from a ActorSpecs protobuf message
        """
        return cls(actor_specs_pb)
