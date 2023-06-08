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

from ..constants import PLAYER_ACTOR_CLASS, ActorSpecType
from .action_space import ActionSpace
from .ndarray_serialization import SerializationFormat
from .observation_space import ObservationSpace
from .spaces_serialization import deserialize_gym_space, serialize_gym_space


class EnvironmentActorSpecs:
    """
    Representation of the specification of an environment within cogment verse for a specific observation space.
    Properties:
        actor_spec:
            The name of the agent role.
        implementation:
            The name of the implementation for the environment.
        num_players:
            Number of players supported by the environment.
        turn_based:
            is this environment turn based (vs real time).
    """

    def __init__(self, environment_specs_pb):
        """
        EnvironmentActorSpecs constructor.
        Shouldn't be called directly, prefer the factory function such as EnvironmentActorSpecs.deserialize or EnvironmentActorSpecs.create.
        """
        self._pb = environment_specs_pb

    def __str__(self):
        result = f"EnvironmentActorSpecs: actor_spec = {self.actor_spec}, implementation = {self.implementation}"
        result += f", num_players = {self.num_players}, turn_based = {self.turn_based}"
        return result

    @property
    def actor_spec(self):
        if hasattr(self._pb, "actor_spec"):
            return ActorSpecType.from_config(self._pb.actor_spec)
        else:
            return ActorSpecType.DEFAULT  # Backwards compatibility with single environment specs setup

    @property
    def implementation(self):
        return self._pb.implementation

    @implementation.setter
    def implementation(self, implementation: str):
        self._pb.implementation = implementation

    @property
    def num_players(self):
        return self._pb.num_players

    @property
    def turn_based(self):
        return self._pb.turn_based

    def get_observation_space(self, render_width=1024):
        """
        Build an instance of the observation space for this environment
        Parameters:
            render_width: optional
                maximum width for the serialized rendered frame in observation
        """
        return ObservationSpace(deserialize_gym_space(self._pb.observation_space), render_width)

    def get_action_space(self, actor_class=PLAYER_ACTOR_CLASS, seed=None):
        """
        Build an instance of the action space for this environment
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
        num_players,
        turn_based,
        observation_space,
        action_space,
        actor_spec=PLAYER_ACTOR_CLASS,
        serilization_format=SerializationFormat.STRUCTURED,
    ):
        """
        Factory function building an homogenous EnvironmentSpecs, ie  with all actors having the same action and observation spaces.
        """
        return cls.deserialize(
            PbEnvironmentSpecs(
                num_players=num_players,
                turn_based=turn_based,
                observation_space=serialize_gym_space(observation_space),
                action_space=serialize_gym_space(action_space),
                actor_spec=actor_spec,
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
        pb_specs = []
        specs_directory = os.path.join(work_dir, "environment_specs", f"{env_name}")

        for file in os.listdir(specs_directory):
            if file.endswith(".yaml"):
                specs_filename = os.path.join(work_dir, "environment_specs", f"{env_name}", file)
                with open(specs_filename, "r", encoding="utf-8") as f:
                    pb_specs.append(ParseDict(yaml.safe_load(f), PbEnvironmentSpecs()))
        return cls.deserialize(pb_specs)

    def save(self, work_dir, env_name):
        """
        Saving to cogment_version work dir cache
        """
        specs_filename = os.path.join(work_dir, "environment_specs", f"{env_name}", f"{self._pb.actor_spec}.yaml")
        os.makedirs(os.path.dirname(specs_filename), exist_ok=True)

        self._pb.implementation = env_name

        with open(specs_filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(MessageToDict(self._pb, preserving_proto_field_name=True), f)
