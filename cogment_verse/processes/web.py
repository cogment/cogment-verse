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
import os

from ..services_directory import ServiceType
from ..web import generate, npm_command, server_main
from .cogment_verse_process import CogmentVerseProcess

log = logging.getLogger(__name__)

WEB_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../web/web_app"))


class WebProcess(CogmentVerseProcess):
    def __init__(self, name, work_dir, specs_filename, services_directory, web_cfg):
        # TODO find a way to detect a rebuild is needed
        if web_cfg.get("build", False):
            log.info("Installing web app dependencies using `npm install`...")
            npm_command(["install", "--no-audit"], WEB_APP_DIR)

            generate(specs_filename, WEB_APP_DIR, True)

            log.info("Building the web app `npm run build`...")
            npm_command(
                ["run", "build"],
                WEB_APP_DIR,
                env={"NODE_ENV": "development" if web_cfg.get("dev", False) else "production"},
            )

        super().__init__(
            name=name,
            target=server_main,
            port=web_cfg.port,
            orchestrator_web_endpoint=services_directory.get(ServiceType.ORCHESTRATOR_WEB_ENDPOINT),
            web_app_dir=WEB_APP_DIR,
            web_components_dir=os.path.join(work_dir, "web_components"),
        )


def create_web_service(work_dir, specs_filename, services_directory, web_cfg):
    return WebProcess("web", work_dir, specs_filename, services_directory, web_cfg)
