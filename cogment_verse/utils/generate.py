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
import subprocess
import sys


def generate(work_dir, specs_filename):
    generate_out_dirname = os.path.join(work_dir, "generate_out")
    os.makedirs(generate_out_dirname, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cogment.generate",
            "--spec",
            specs_filename,
            "--output",
            os.path.join(generate_out_dirname, "cog_settings.py"),
        ],
        cwd=os.path.dirname(specs_filename),
        check=True,
    )
    sys.path.append(generate_out_dirname)
