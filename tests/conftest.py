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

from cogment_verse.utils.generate import generate

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

generate(
    work_dir=os.path.join(ROOT_DIR, ".cogment_verse"),
    specs_filename=os.path.join(ROOT_DIR, "cogment_verse/specs/cogment.yaml"),
)
