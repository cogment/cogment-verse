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

import os

from google.protobuf.json_format import MessageToDict, ParseDict
import yaml

from data_pb2 import EnvironmentSpecs  # pylint: disable=import-error


def save_environment_specs(work_dir, env_name, env_specs):
    specs_filename = os.path.join(work_dir, "environment_specs", f"{env_name}.yaml")
    os.makedirs(os.path.dirname(specs_filename), exist_ok=True)

    env_specs.implementation = env_name

    with open(specs_filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(MessageToDict(env_specs, preserving_proto_field_name=True), f)


def load_environment_specs(work_dir, env_name):
    specs_filename = os.path.join(work_dir, "environment_specs", f"{env_name}.yaml")

    with open(specs_filename, "r", encoding="utf-8") as f:
        return ParseDict(yaml.safe_load(f), EnvironmentSpecs())
