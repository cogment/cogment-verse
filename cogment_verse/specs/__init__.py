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

import cog_settings  # pylint: disable=import-error
from data_pb2 import AgentConfig, EnvironmentConfig  # pylint: disable=import-error

from cogment_verse.constants import (
    HUMAN_ACTOR_IMPL,
    OBSERVER_ACTOR_CLASS,
    PLAYER_ACTOR_CLASS,
    TEACHER_ACTOR_CLASS,
    WEB_ACTOR_NAME,
)

from .encode_rendered_frame import encode_rendered_frame
from .environment_specs import EnvironmentSpecs
