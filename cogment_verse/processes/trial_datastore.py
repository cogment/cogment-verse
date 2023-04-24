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

from cogment_verse.constants import DATASTORE_DIR

from ..services_directory import ServiceType
from .cogment_cli_process import CogmentCliProcess

log = logging.getLogger(__name__)


def create_trial_datastore_service(work_dir, trial_datastore_cfg, services_directory):

    data_dir = os.path.join(work_dir, DATASTORE_DIR)
    os.makedirs(data_dir, exist_ok=True)

    port = trial_datastore_cfg.port

    services_directory.add(
        service_type=ServiceType.TRIAL_DATASTORE,
        service_endpoint=f"grpc://localhost:{port}",
    )

    cli_args = [
        "services",
        "trial_datastore",
        "--log_format=json",
        f"--log_level={trial_datastore_cfg.log_level}",
        f"--port={port}",
    ]

    if trial_datastore_cfg.local_file_storage:
        cli_args.append(f"--file_storage=.cogment_verse/{ServiceType.TRIAL_DATASTORE.value}/trial_datastore.db")
        log.info(f"Trial Datastore local file storage enabled.")

    log.info(f"Trial Datastore starting on port [{port}]...")

    return CogmentCliProcess(
        name="trial_datastore",
        work_dir=work_dir,
        cli_args=cli_args,
    )

