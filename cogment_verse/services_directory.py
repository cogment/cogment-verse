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

from enum import Enum
import inspect
from random import choice
import cogment


class ServiceType(Enum):
    ORCHESTRATOR = "orchestrator"
    ORCHESTRATOR_WEB_ENDPOINT = "orchestrator_web_endpoint"
    ENVIRONMENT = "environment"
    ACTOR = "actor"
    TRIAL_DATASTORE = "trial_datastore"
    MODEL_REGISTRY = "model_registry"
    WEB = "web"


class ServiceDirectory:
    def __init__(self):
        self._directory = {}

    def add(self, service_type, service_endpoint, service_name=None):
        if service_type.value not in self._directory:
            self._directory[service_type.value] = {}

        if service_name not in self._directory[service_type.value]:
            self._directory[service_type.value][service_name] = []

        self._directory[service_type.value][service_name].append(service_endpoint)

    def get(self, service_type, service_name=None):
        if service_type.value not in self._directory:
            raise RuntimeError(f"No service of type [{service_type.value}] registered")

        if service_name not in self._directory[service_type.value]:
            raise RuntimeError(f"No service named [{service_name}] of type [{service_type.value}] registered")

        return choice(self._directory[service_type.value][service_name])

    def get_service_names(self, service_type):
        if service_type.value not in self._directory:
            return []

        return [*self._directory[service_type.value].keys()]

    async def get_datastore(self, context):
        datastore = context.get_datastore(endpoint=cogment.Endpoint(self.get(ServiceType.TRIAL_DATASTORE)))
        if inspect.isawaitable(datastore):
            return await datastore
        return datastore

    async def get_controller(self, context):
        controller = context.get_controller(endpoint=cogment.Endpoint(self.get(ServiceType.ORCHESTRATOR)))
        if inspect.isawaitable(controller):
            return await controller
        return controller

    async def get_model_registry(self, context):
        registry = context.get_model_registry_v2(endpoint=cogment.Endpoint(self.get(ServiceType.MODEL_REGISTRY)))
        if inspect.isawaitable(registry):
            return await registry
        return registry
