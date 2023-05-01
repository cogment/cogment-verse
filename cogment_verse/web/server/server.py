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

import json
import logging
import os

import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

log = logging.getLogger(__name__)

WEB_CFG_ENV_VAR = "COGMENT_VERSE_ORCHESTRATOR_WEB_CFG"


def create_app():
    web_app_cfg = json.loads(os.getenv(WEB_CFG_ENV_VAR))

    with open(os.path.join(web_app_cfg["served_dir"], "index.html"), encoding="utf-8") as homepage_file:
        homepage_content = homepage_file.read()

    orchestrator_web_endpoint = web_app_cfg["orchestrator_web_endpoint"]
    homepage_content = homepage_content.replace(
        'ORCHESTRATOR_WEB_ENDPOINT = ""', f'ORCHESTRATOR_WEB_ENDPOINT = "{orchestrator_web_endpoint}"'
    )

    async def homepage(_request):
        return HTMLResponse(homepage_content)

    return Starlette(
        routes=[
            Mount("/static", app=StaticFiles(directory=os.path.join(web_app_cfg["served_dir"], "static"), html=True)),
            Mount("/assets", app=StaticFiles(directory=os.path.join(web_app_cfg["served_dir"], "assets"), html=True)),
            Mount("/components", app=StaticFiles(directory=web_app_cfg["web_components_dir"], check_dir=False)),
            Route("/{rest_of_path:path}", endpoint=homepage),
        ]
    )


def server_main(
    name,  # pylint: disable=unused-argument
    on_ready,
    orchestrator_web_endpoint,
    port,
    served_dir,
    web_components_dir,
):
    # Writing the configuration for the web app in an environment variable
    web_app_cfg = {
        "orchestrator_web_endpoint": orchestrator_web_endpoint,
        "served_dir": served_dir,
        "web_components_dir": web_components_dir
    }
    os.environ[WEB_CFG_ENV_VAR] = json.dumps(web_app_cfg)

    on_ready()

    log.info(f"Starting production web server on port [{port}]")
    uvicorn.run(
        "cogment_verse.web.server.server:create_app",
        factory=True,
        host="0.0.0.0",
        port=port,
        log_level="error",
    )
