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
from typing import List, Optional

import yaml
from cogment_verse.specs.ndarray_serialization import SerializationFormat
from data_pb2 import EnvironmentSpecs as PbEnvironmentSpecs
from data_pb2 import MultiEnvironmentSpecs as PbMultiEnvironmentSpecs  # pylint: disable=import-error
from google.protobuf.json_format import MessageToDict, ParseDict

from cogment_verse.specs.single_environment_specs import EnvironmentSpecs

from ..constants import ActorSpecType

log = logging.getLogger(__name__)


class EnvironmentSpecs:
    """ Representation of the specification of an environment within cogment verse for multiple observation spaces.
    """

    def __init__(self, environment_specs: List[EnvironmentSpecs] = []):
        self._specs = {}

        if environment_specs:
            for spec in environment_specs:
                self._specs[spec.actor_spec] = spec
        else:  # Backward compatibility with single environment specs setup.
            self._specs[ActorSpecType.DEFAULT] = environment_specs

    def __getitem__(self, actor_spec: ActorSpecType):
        if actor_spec in self._specs:
            return self._specs[actor_spec]
        else:
            raise ValueError(f"Actor spec type ({actor_spec.value}) is not added to the environment_specs: [{','.join([spec_type.value for spec_type in self._specs.keys()])}]")

    def __add__(self, spec: EnvironmentSpecs):
        if spec.actor_spec not in self._specs:
            self._specs[spec.actor_spec] = spec

    def remove(self, spec):
        self._specs.pop(spec.actor_spec, None)

    def __len__(self):
        return len(self._specs)

    def __str__(self):
        return f"EnvrionmentSpecs: [{', '.join([str(spec) for spec_type, spec in self._specs.items()])}]"

    @property
    def implementation(self) -> Optional[str]:
        for _, spec in self._specs.items():
            return spec.implementation
        return None

    @implementation.setter
    def implementation(self, implementation: str):
        for _, spec in self._specs.items():
            spec.implementation = implementation

    @property
    def num_players(self) -> int:
        return sum([spec.num_players for _, spec in self._specs.items()])

    def serialize(self):
        """
        Serialize to a MultiEnvironmentSpecs protobuf message
        """
        return PbMultiEnvironmentSpecs(environment_specs=[spec.serialize() for spec_type, spec in self._specs.items()])

    @classmethod
    def create(
        cls,
        num_players,
        turn_based,
        observation_space,
        action_space,
        actor_spec=ActorSpecType.DEFAULT,
        serilization_format=SerializationFormat.STRUCTURED,
    ):
        """
        Factory function building an homogenous EnvironmentSpecs, ie  with all actors having the same action and observation spaces.
        """
        return cls([EnvironmentSpecs.create(num_players, turn_based, observation_space, action_space, actor_spec, serilization_format,)])

    @classmethod
    def deserialize(cls, specs_pb: PbMultiEnvironmentSpecs):
        """
        Factory function building a MultiEnvironmentSpecs instance from a MultiEnvironmentSpecs protobuf message or
        a list of EnvironmentSpecs protobuf message.
        """
        spec_list = []
        print(type(specs_pb))
        for spec_pb in specs_pb.environment_specs:
            spec_list.append(EnvironmentSpecs.deserialize(spec_pb))
        return cls(spec_list)

    @classmethod
    def load(cls, work_dir, env_name):
        """
        Factory function building an EnvironmentSpecs from cogment_version work dir cache
        """
        spec_list = []
        specs_directory = os.path.join(work_dir, "environment_specs", f"{env_name}")

        for file in os.listdir(specs_directory):
            if file.endswith(".yaml"):
                specs_filename = os.path.join(specs_directory, file)
                with open(specs_filename, "r", encoding="utf-8") as f:
                    spec = EnvironmentSpecs.deserialize(ParseDict(yaml.safe_load(f), PbEnvironmentSpecs()))
                    print(f".load spec: {spec}")
                    spec_list.append(spec)

        return cls(spec_list)

    def save(self, work_dir, env_name):
        """
        Saving to cogment_version work dir cache
        """
        for actor_spec, spec in self._specs.items():
            print(f"actor_spec: {actor_spec}, {actor_spec.value}")
            specs_filename = os.path.join(work_dir, "environment_specs", f"{env_name}", f"{actor_spec.value}.yaml")
            print(f".save specs_filename: {specs_filename}")
            os.makedirs(os.path.dirname(specs_filename), exist_ok=True)

            self.implementation = env_name

            with open(specs_filename, "w", encoding="utf-8") as f:
                yaml.safe_dump(MessageToDict(spec._pb, preserving_proto_field_name=True), f)
