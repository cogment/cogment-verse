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

COGMENT_VERSION = "v2.12.1"
WEB_ACTOR_NAME = "web_actor"  # At the moment the web client can only join the trial using the actor name which means a unique name is required
HUMAN_ACTOR_IMPL = "client"

TEACHER_ACTOR_CLASS = "teacher"
PLAYER_ACTOR_CLASS = "player"
OBSERVER_ACTOR_CLASS = "observer"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WORK_DIR = os.path.join(ROOT_DIR, ".cogment_verse")
DATASTORE_DIR = "trial_datastore"
MODEL_REGISTRY_DIR = "model_registry"

DEFAULT_CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DEFAULT_CONFIG_NAME = "config"

TEST_DIR = os.path.join(ROOT_DIR, "tests")
FUNCTIONAL_TEST_DIR = os.path.join(TEST_DIR, "functional")
