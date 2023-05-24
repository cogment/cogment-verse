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

from ..services_directory import ServiceType
from .cogment_cli_process import CogmentCliProcess


def create_orchestrator_service(work_dir, orchestrator_cfg, services_directory):
    port = orchestrator_cfg.port
    web_port = orchestrator_cfg.web_port
    web_endpoint = orchestrator_cfg.web_endpoint
    services_directory.add(
        service_type=ServiceType.ORCHESTRATOR,
        service_endpoint=f"grpc://localhost:{port}",
    )
    services_directory.add(
        service_type=ServiceType.ORCHESTRATOR_WEB_ENDPOINT,
        service_endpoint=web_endpoint,
    )
    return CogmentCliProcess(
        name="orchestrator",
        work_dir=work_dir,
        cli_args=[
            "services",
            "orchestrator",
            "--log_format=json",
            f"--log_level={orchestrator_cfg.log_level}",
            f"--actor_port={port}",
            f"--lifecycle_port={port}",
            f"--actor_web_port={web_port}",
        ],
    )
