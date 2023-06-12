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

import logging
import shutil
import subprocess
import os

NPM_BIN = shutil.which("npm")

log = logging.getLogger(__name__)


def npm_command(args, cwd, env=None):
    try:
        args = [NPM_BIN, *args]
        env = {
            **os.environ,
            **(env if env is not None else {}),
        }
        res = subprocess.run(args, cwd=cwd, capture_output=True, check=True, env=env)
        log.debug(f"Call to {args} returned {res.stdout.decode('utf-8')}")
    except subprocess.CalledProcessError as err:
        log.error(
            f"Error while running [{args}] in [{cwd}]\n---STDOUT---\n{err.stdout.decode('utf-8')}\n---STDERR---\n{err.stderr.decode('utf-8')}"
        )
        raise RuntimeError(f"Error while running [{args}] in [{cwd}]") from err
