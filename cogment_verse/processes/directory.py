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
from urllib.parse import urlparse

from .cogment_cli_process import CogmentCliProcess
from ..services_directory import ServiceType

log = logging.getLogger(__name__)


def create_directory_service(work_dir, directory_cfg, services_directory):
    port = directory_cfg.port

    services_directory.add(
        service_type=ServiceType.DIRECTORY,
        service_endpoint=f"grpc://localhost:{port}",
    )

    log.info(f"Directory service starting on port [{port}]...")

    return CogmentCliProcess(
        name="directory",
        work_dir=work_dir,
        cli_args=[
            "services",
            "directory",
            "--log_format=json",
            f"--log_level={directory_cfg.log_level}",
            f"--port={port}",
            f"--registration_lag={directory_cfg.registration_lag}"
        ],
    )

