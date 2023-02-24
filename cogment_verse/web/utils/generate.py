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

import logging
import os
import shutil


from .npm import npm_command

log = logging.getLogger(__name__)


def generate(specs_filename, web_dir, force=False):
    cog_generate_sample_out_file = os.path.join(web_dir, "src/CogSettings.d.ts")
    cog_generate_source_files = [
        specs_filename,
        os.path.join(os.path.dirname(specs_filename), "data.proto"),
        os.path.join(os.path.dirname(specs_filename), "ndarray.proto"),
        os.path.join(os.path.dirname(specs_filename), "spaces.proto"),
    ]  # TODO make that more generic

    do_generate = force

    if not do_generate and not os.path.isfile(cog_generate_sample_out_file):
        do_generate = True

    if not do_generate:
        cog_generate_output_mtime = os.stat(cog_generate_sample_out_file).st_mtime
        do_generate = any(
            os.stat(source_file).st_mtime > cog_generate_output_mtime for source_file in cog_generate_source_files
        )

    if do_generate:
        for source_file in cog_generate_source_files:
            shutil.copy(source_file, web_dir)
        log.info("Running code generation for the web components using `npm run cogment_generate`...")
        npm_command(["run", "cogment_generate"], web_dir)
    else:
        log.info("Nothing to do for the web components code generation")
