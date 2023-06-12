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

import json
import logging
import os

import uvicorn
from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

log = logging.getLogger(__name__)

WEB_CFG_ENV_VAR = "COGMENT_VERSE_ORCHESTRATOR_WEB_CFG"


def create_app():
    web_app_cfg = json.loads(os.getenv(WEB_CFG_ENV_VAR))

    templates = Jinja2Templates(directory=web_app_cfg["web_app_dir"])

    async def homepage(request):
        return templates.TemplateResponse(
            "index.html", {"request": request, "orchestrator_web_endpoint": web_app_cfg["orchestrator_web_endpoint"]}
        )

    return Starlette(
        routes=[
            Mount(
                "/dist",
                app=StaticFiles(directory=os.path.join(web_app_cfg["web_app_dir"], "dist"), html=True),
                name="dist",
            ),
            Mount(
                "/assets",
                app=StaticFiles(directory=os.path.join(web_app_cfg["web_app_dir"], "assets"), html=True),
                name="assets",
            ),
            Mount(
                "/components",
                app=StaticFiles(directory=web_app_cfg["web_components_dir"], check_dir=False),
                name="components",
            ),
            Route("/{rest_of_path:path}", endpoint=homepage),
        ]
    )


def server_main(
    name,  # pylint: disable=unused-argument
    on_ready,
    orchestrator_web_endpoint,
    port,
    web_app_dir,
    web_components_dir,
):
    # Writing the configuration for the web app in an environment variable
    web_app_cfg = {
        "orchestrator_web_endpoint": orchestrator_web_endpoint,
        "web_app_dir": web_app_dir,
        "web_components_dir": web_components_dir,
    }
    os.environ[WEB_CFG_ENV_VAR] = json.dumps(web_app_cfg)

    on_ready()

    log.info(f"Starting production web server on port [{port}]")
    uvicorn.run(
        "cogment_verse.web.server:create_app",
        factory=True,
        host="0.0.0.0",
        port=port,
        log_level="error",
    )
