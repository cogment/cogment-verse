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

from ..utils.npm import NPM_BIN


def create_dev_server_popen_kwargs(port, orchestrator_web_endpoint, **kwargs):
    extended_env = os.environ.copy()
    extended_env |= {
        "PORT": f"{port}",
        "BROWSER": "none",  # Don't automatically open a browser
        "REACT_APP_ORCHESTRATOR_WEB_ENDPOINT": orchestrator_web_endpoint,
    }

    return {
        **kwargs,
        "args": [NPM_BIN, "run", "start"],
        "env": extended_env,
        "cwd": os.path.abspath(os.path.join(os.path.dirname(__file__), "../web_app")),
    }
