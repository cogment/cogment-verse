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

from .cogment_verse_process import CogmentVerseProcess
from .orchestrator import create_orchestrator_service
from .environment import create_environment_service
from .actor import create_actor_service
from .run import create_run_process
from .trial_datastore import create_trial_datastore_service
from .model_registry import create_model_registry_service
from .web import create_web_service
