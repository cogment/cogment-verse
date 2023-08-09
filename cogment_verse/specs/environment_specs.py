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

from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import yaml
from cogment_verse.specs.actor_specs import ActorSpecs
from cogment_verse.specs.ndarray_serialization import SerializationFormat
from cogment_verse.specs.spaces_serialization import serialize_space
from data_pb2 import EnvironmentSpecs as PbEnvironmentSpecs  # pylint: disable=import-error
from data_pb2 import ActorSpecs as PbActorSpecs  # pylint: disable=import-error
from google.protobuf.json_format import MessageToDict, ParseDict

from ..constants import PLAYER_ACTOR_CLASS, ActorSpecType

log = logging.getLogger(__name__)


class EnvironmentSpecs:
    """ Representation of the specification of an environment within cogment verse for multiple observation spaces.
    """

    # def __init__(self, num_players, turn_based, actor_specs: List[ActorSpecs] = []):

    def __init__(self, pb):
        self._pb = pb
        self._actor_specs = {}

        if pb.actor_specs:
            for spec_pb in pb.actor_specs:
                self._actor_specs[spec_pb.spec_type] = ActorSpecs.deserialize(spec_pb)
        else:
            # TODO: raise error
            pass
            # self._actor_specs[ActorSpecType.DEFAULT] = actor_specs

    def __getitem__(self, spec_type: Union[ActorSpecType, str]) -> ActorSpecs:
        if isinstance(spec_type, ActorSpecType):
            spec_type = spec_type.value

        if spec_type in self._actor_specs:
            return self._actor_specs[spec_type]
        else:
            raise ValueError(f"Actor specs type ({spec_type}) is not in the environment specs types: [{', '.join([spec_type for spec_type in self._actor_specs.keys()])}]")

    def __add__(self, actor_specs: ActorSpecs):
        if actor_specs.spec_type.value not in self._actor_specs:
            self._actor_specs[actor_specs.spec_type.value] = actor_specs

    # def remove(self, spec):
    #     self._actor_specs.pop(spec.actor_spec, None)

    def __len__(self):
        return len(self._actor_specs)

    def __str__(self):
        return f"EnvrionmentSpecs: [{', '.join([str(spec) for spec_type, spec in self._actor_specs.items()])}]"

    @property
    def implementation(self) -> Optional[str]:
        return self._pb.implementation

    @property
    def num_players(self) -> int:
        return self._pb.num_players

    @property
    def turn_based(self):
        return self._pb.turn_based

    def serialize(self):
        """
        Serialize to a EnvironmentSpecs protobuf message
        """
        return PbEnvironmentSpecs(
            implementation=self.implementation,
            turn_based=self.turn_based,
            num_players=self.num_players,
            actor_specs=[actor_specs.serialize() for _, actor_specs in self._actor_specs.items()]
        )

    @classmethod
    def create_heterogeneous(
        cls,
        num_players,
        turn_based,
        actor_specs: List[ActorSpecs] = [],
    ):
        return cls.deserialize(PbEnvironmentSpecs(
            num_players=num_players,
            turn_based=turn_based,
            actor_specs=[spec.serialize() for spec in actor_specs],
        ))


    @classmethod
    def create_homogeneous(
        cls,
        num_players,
        turn_based,
        observation_space,
        action_space,
        web_components_file=None,
        actor_class=PLAYER_ACTOR_CLASS,
        serialization_format=SerializationFormat.STRUCTURED,
    ):
        """
        Factory function building an homogenous EnvironmentSpecs, ie  with all actors having the same action and observation spaces.
        """
        return cls.deserialize(PbEnvironmentSpecs(
            num_players=num_players,
            turn_based=turn_based,
            actor_specs=[PbActorSpecs(
                spec_type=ActorSpecType.DEFAULT.value,
                observation_space=serialize_space(observation_space, serialization_format),
                action_space=serialize_space(action_space, serialization_format),
                web_components_file=web_components_file,
            )],
        ))

    @classmethod
    def deserialize(cls, environment_specs_pb: PbEnvironmentSpecs):
        """
        Factory function building a EnvironmentSpecs instance from an EnvironmentSpecs protobuf message.
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
