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

from cogment_verse.constants import MODEL_REGISTRY_DIR

from ..services_directory import ServiceType
from .cogment_cli_process import CogmentCliProcess

log = logging.getLogger(__name__)


def create_model_registry_service(work_dir, model_registry_cfg, services_directory):
    model_registry_data_dir = os.path.join(work_dir, MODEL_REGISTRY_DIR)
    os.makedirs(model_registry_data_dir, exist_ok=True)

    port = model_registry_cfg.port

    services_directory.add(
        service_type=ServiceType.MODEL_REGISTRY,
        service_endpoint=f"grpc://localhost:{port}",
    )

    log.info(f"Model Registry starting on port [{port}]...")

    return CogmentCliProcess(
        name="model_registry",
        work_dir=work_dir,
        cli_args=[
            "services",
            "model_registry",
            "--log_format=json",
            f"--log_level={model_registry_cfg.log_level}",
            f"--archive_dir={model_registry_data_dir}",
            f"--port={port}",
            f"--sent_version_chunk_size={1024 * 1024 * 2}",  # TODO make those options configurable
            "--cache_max_items=100",
        ],
    )
