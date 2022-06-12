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

from data_pb2 import (  # pylint: disable=import-error
    AgentConfig,
    EnvironmentConfig,
    EnvironmentSpecs,
    Observation,
    PlayerAction,
    Space,
    SpaceValue,
)
import cog_settings  # pylint: disable=import-error
from cogment_verse.constants import (
    WEB_ACTOR_NAME,
    HUMAN_ACTOR_IMPL,
    TEACHER_ACTOR_CLASS,
    PLAYER_ACTOR_CLASS,
    OBSERVER_ACTOR_CLASS,
)

from .environment_specs import save_environment_specs, load_environment_specs
from .encode_rendered_frame import encode_rendered_frame
from .ndarray import deserialize_ndarray, serialize_ndarray
from .sample_space import sample_space
from .flatten import flattened_dimensions, flatten, unflatten
