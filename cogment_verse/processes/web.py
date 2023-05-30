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
import time

from ..services_directory import ServiceType
from ..web import create_dev_server_popen_kwargs, generate, npm_command, server_main
from .cogment_verse_process import CogmentVerseProcess
from .popen_process import PopenProcess

log = logging.getLogger(__name__)

WEB_SOURCES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../web/web_app"))

WEB_BUILD_DIR = os.path.abspath(os.path.join(WEB_SOURCES_DIR, "build"))


def on_cra_log(_name, log_line):
    log.info(log_line)


def on_awaiting_cra_ready():
    time.sleep(5)  # TODO find a better way to do that ?


class NpmProcess(PopenProcess):
    def __init__(self, name, specs_filename, services_directory, web_cfg):
        log.info("Installing web components dependencies using `npm install`...")
        npm_command(["install", "--no-audit"], WEB_SOURCES_DIR)

        generate(specs_filename, WEB_SOURCES_DIR, web_cfg.get("force_rebuild", False))

        super().__init__(
            name=name,
            on_log=on_cra_log,
            on_awaiting_ready=on_awaiting_cra_ready,
            **create_dev_server_popen_kwargs(
                port=web_cfg.port,
                orchestrator_web_endpoint=services_directory.get(ServiceType.ORCHESTRATOR_WEB_ENDPOINT),
            ),
        )


class WebProcess(CogmentVerseProcess):
    def __init__(self, name, specs_filename, services_directory, web_cfg):
        # TODO find a better way to detect a rebuild is needed
        if not os.path.isdir(WEB_BUILD_DIR) or web_cfg.get("build", False):
            log.info("Installing web app dependencies using `npm install`...")
            npm_command(["install", "--no-audit", "--install-links"], WEB_SOURCES_DIR)

            generate(specs_filename, WEB_SOURCES_DIR, True)

            log.info("Building the web app `npm run build`...")
            npm_command(["run", "build"], WEB_SOURCES_DIR)

        super().__init__(
            name=name,
            target=server_main,
            port=web_cfg.port,
            orchestrator_web_endpoint=services_directory.get(ServiceType.ORCHESTRATOR_WEB_ENDPOINT),
            served_dir=WEB_BUILD_DIR,
        )


def create_web_service(specs_filename, services_directory, web_cfg):
    if web_cfg.get("dev", False):
        return NpmProcess("web", specs_filename, services_directory, web_cfg)
    return WebProcess("web", specs_filename, services_directory, web_cfg)
