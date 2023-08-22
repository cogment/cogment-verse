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
from data_pb2 import EnvironmentSpecs as PbEnvironmentSpecs  # pylint: disable=import-error
from google.protobuf.json_format import MessageToDict, ParseDict

from ..constants import DEFAULT_RENDERED_WIDTH, PLAYER_ACTOR_CLASS
from .action_space import ActionSpace
from .ndarray_serialization import SerializationFormat
from .observation_space import ObservationSpace
from .spaces_serialization import deserialize_space, serialize_gym_space


class EnvironmentSpecs:
    """
    Representation of the specification of an environment within Cogment Verse

    Properties:
        implementation:
            The name of the implementation for the environment.
        num_players:
            Number of players supported by the environment.
        turn_based:
            is this environment turn based (vs real time).
    """

    def __init__(self, environment_specs_pb):
        """
        EnvironmentSpecs constructor.
        Shouldn't be called directly, prefer the factory function such as EnvironmentSpecs.deserialize or EnvironmentSpecs.create_homogeneous.
        """
        self._pb = environment_specs_pb

    @property
    def implementation(self):
        return self._pb.implementation

    @property
    def num_players(self):
        return self._pb.num_players

    @property
    def turn_based(self):
        return self._pb.turn_based

    @property
    def web_components_file(self):
        return self._pb.web_components_file

    def get_observation_space(self, render_width=DEFAULT_RENDERED_WIDTH):
        """
        Build an instance of the observation space for this environment

        Parameters:
            render_width: optional
                maximum width for the serialized rendered frame in observation

        NOTE: In the future we'll want to support different observation space per agent role
        """
        return ObservationSpace(deserialize_space(self._pb.observation_space), render_width)

    def get_action_space(self, actor_class=PLAYER_ACTOR_CLASS, seed=None):
        """
        Build an instance of the action space for this environment

        Parameters:
            actor_class: optional
                the class of the actor for which we want to retrieve the action space.
                this parameters is mostly useful when serializing actions.
            seed: optional
                the seed used when generating random actions

        NOTE: In the future we'll want to support different action space per agent roles
        """
        return ActionSpace(deserialize_space(self._pb.action_space), actor_class, seed)

    @classmethod
    def create_homogeneous(
        cls,
        num_players,
        turn_based,
        observation_space,
        action_space,
        web_components_file=None,
        serialization_format=SerializationFormat.STRUCTURED,
    ):
        """
        Factory function building an homogenous EnvironmentSpecs, ie  with all actors having the same action and observation spaces.
        """
        return cls.deserialize(
            PbEnvironmentSpecs(
                num_players=num_players,
                turn_based=turn_based,
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
    def deserialize(cls, environment_specs_pb):
        """
        Factory function building an EnvironmentSpecs instance from a EnvironmentSpecs protobuf message
        """
        return cls(environment_specs_pb)

    @classmethod
    def load(cls, work_dir, env_name):
        """
        Factory function building an EnvironmentSpecs from cogment_version work dir cache
        """

        specs_filename = os.path.join(work_dir, "environment_specs", f"{env_name}.yaml")

        with open(specs_filename, "r", encoding="utf-8") as f:
            return cls.deserialize(ParseDict(yaml.safe_load(f), PbEnvironmentSpecs()))

    def save(self, work_dir, env_name):
        """
        Saving to cogment_version work dir cache
        """

        specs_filename = os.path.join(work_dir, "environment_specs", f"{env_name}.yaml")
        os.makedirs(os.path.dirname(specs_filename), exist_ok=True)

        self._pb.implementation = env_name

        with open(specs_filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(MessageToDict(self._pb, preserving_proto_field_name=True), f)
