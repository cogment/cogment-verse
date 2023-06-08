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

from names_generator import generate_name
from omegaconf import OmegaConf

from .constants import DEFAULT_WORK_DIR, HUMAN_ACTOR_IMPL
from .processes import (
    create_actor_service,
    create_environment_service,
    create_model_registry_service,
    create_orchestrator_service,
    create_run_process,
    create_trial_datastore_service,
    create_web_service,
)
from .services_directory import ServiceDirectory, ServiceType
from .utils.find_free_port import find_free_port

log = logging.getLogger(__name__)

SPEC_FILEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "specs", "cogment.yaml"))


def register_generate_port_resolver():
    ports = {}

    def generate_port(key):
        nonlocal ports
        if key in ports:
            return ports[key]
        port = find_free_port()
        ports[key] = port
        return port

    OmegaConf.register_new_resolver("generate_port", generate_port)


def register_generate_name_resolver():
    names = {}

    def do_generate_name(key):
        nonlocal names
        if key in names:
            return names[key]
        name = generate_name()
        names[key] = name
        return name

    OmegaConf.register_new_resolver("generate_name", do_generate_name)


register_generate_port_resolver()
register_generate_name_resolver()


class App:
    def __init__(self, cfg, work_dir=DEFAULT_WORK_DIR):
        self.cfg = OmegaConf.create(cfg)
        OmegaConf.resolve(
            cfg
        )  # The configuration (or sub configurations) will be passed to other processes let's avoid surprises
        self.services_directory = ServiceDirectory()
        self.services_process = []

        for service_type, services_cfg in cfg.services.items():
            if service_type not in [service_type.value for service_type in ServiceType]:
                raise RuntimeError(f"Unknown service type [{service_type}]")
            if service_type == ServiceType.ORCHESTRATOR.value:
                orchestrator_service_cfg = services_cfg
                self.services_process.append(
                    create_orchestrator_service(work_dir, orchestrator_service_cfg, self.services_directory)
                )
            elif service_type == ServiceType.TRIAL_DATASTORE.value:
                trial_datastore_service_cfg = services_cfg
                self.services_process.append(
                    create_trial_datastore_service(work_dir, trial_datastore_service_cfg, self.services_directory)
                )
            elif service_type == ServiceType.MODEL_REGISTRY.value:
                model_registry_service_cfg = services_cfg
                self.services_process.append(
                    create_model_registry_service(work_dir, model_registry_service_cfg, self.services_directory)
                )
            elif service_type == ServiceType.ENVIRONMENT.value:
                environment_service_cfg = services_cfg
                self.services_process.append(
                    create_environment_service(
                        work_dir, SPEC_FILEPATH, environment_service_cfg, self.services_directory
                    )
                )
            elif service_type == ServiceType.ACTOR.value:
                for actor_service_cfg in services_cfg.values():
                    self.services_process.append(
                        create_actor_service(work_dir, SPEC_FILEPATH, actor_service_cfg, self.services_directory)
                    )
            elif service_type == ServiceType.WEB.value:
                self.services_process.append(
                    create_web_service(work_dir, SPEC_FILEPATH, self.services_directory, services_cfg)
                )
                self.services_directory.add(
                    service_type=ServiceType.ACTOR,
                    service_name=HUMAN_ACTOR_IMPL,
                    service_endpoint="cogment://client",
                )
            else:
                raise NotImplementedError()

        run_cfg = cfg.get("run", None)
        if run_cfg is not None:
            self.run_process = create_run_process(work_dir, SPEC_FILEPATH, self.services_directory, run_cfg)
        else:
            self.run_process = None

    def start(self):
        log.info("Start services...")
        for service_process in self.services_process:
            service_process.start()

        for service_process in self.services_process:
            service_process.await_ready()
        log.info("Services ready")

        if self.run_process:
            log.info("Run starting...")
            self.run_process.start()

    def terminate(self):
        if self.run_process:
            self.run_process.terminate()
        # TODO terminated signal ?

        log.info("Terminating services...")
        for service_process in reversed(self.services_process):
            service_process.terminate()

    def join(self):
        try:
            if self.run_process:
                self.run_process.join()
                log.info("Run ended")

                # Run ended, terminate the services
                log.info("Terminating services...")
                for service_process in reversed(self.services_process):
                    service_process.terminate()

            for service_process in self.services_process:
                service_process.join()
            log.info("Services ended")
        except KeyboardInterrupt:
            log.warning("interrupted by user")
