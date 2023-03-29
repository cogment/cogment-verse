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
import multiprocessing as mp
import hydra

import cogment_verse
from cogment_verse.services_directory import ServiceType
from main import CONFIG_DIR, DEFAULT_WORK_DIR

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="launch_local_services")
def launch_local_services(cfg, work_dir=None):

    if not work_dir:
        work_dir = os.environ.get("COGMENT_VERSE_WORK_DIR", DEFAULT_WORK_DIR)
    datastore_dir = os.path.join(work_dir, ServiceType.TRIAL_DATASTORE.value)
    model_registry_dir = os.path.join(work_dir, ServiceType.MODEL_REGISTRY.value)

    os.makedirs(datastore_dir, exist_ok=True)
    os.makedirs(model_registry_dir, exist_ok=True)
    app = cogment_verse.App(cfg, work_dir=work_dir)

    trial_datastore_process = None
    model_registry_process = None

    for service_process in app.services_process:
        if service_process.name == ServiceType.TRIAL_DATASTORE.value:
            trial_datastore_process = service_process
        if service_process.name == ServiceType.ORCHESTRATOR.value:
            orchestrator_process = service_process
        if service_process.name == ServiceType.MODEL_REGISTRY.value:
            model_registry_process = service_process

    try:
        log.info("Starting Trial Datastore.")
        orchestrator_process.start()
        orchestrator_process.await_ready()

        trial_datastore_process.start()
        trial_datastore_process.await_ready()
        log.info("Trial Datastore ready.")

        log.info("Starting Model Registry.")
        model_registry_process.start()
        model_registry_process.await_ready()
        log.info("Model Registry ready.")

    except KeyboardInterrupt:
        orchestrator_process.join()
        trial_datastore_process.join()
        log.info("Trial Datastore closed.")
        model_registry_process.join()
        log.info("Model Registry closed.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    launch_local_services()
