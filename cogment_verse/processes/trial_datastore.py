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

from .cogment_cli_process import CogmentCliProcess
from ..services_directory import ServiceType


def create_trial_datastore_service(work_dir, trial_datastore_cfg, services_directory):
    # TODO support other options

    port = trial_datastore_cfg.port

    services_directory.add(
        service_type=ServiceType.TRIAL_DATASTORE,
        service_endpoint=f"grpc://localhost:{port}",
    )

    return CogmentCliProcess(
        name="trial_datastore",
        work_dir=work_dir,
        cli_args=[
            "services",
            "trial_datastore",
            "--log_format=json",
            f"--log_level={trial_datastore_cfg.log_level}",
            f"--port={port}",
        ],
    )
