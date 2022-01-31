# Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import asyncio
import copy
import logging
import random

import cogment
from cogment_verse.api.run_api_pb2 import DESCRIPTOR as RUN_DESCRIPTOR
from cogment_verse.api.run_api_pb2_grpc import add_RunServicer_to_server
from cogment_verse.model_registry_client import ModelRegistryClient
from cogment_verse.run.run_servicer import RunServicer
from cogment_verse.run.run_session import RunSession
from cogment_verse.trial_datastore_client import TrialDatastoreClient
from cogment_verse.utils import LRU
from grpc_reflection.v1alpha import reflection
from prometheus_client.core import REGISTRY

log = logging.getLogger(__name__)

# pylint: disable=too-many-arguments


def set_config_index(config, actor_idx):
    config.actor_index = actor_idx
    return config


# RunContext holds the context information to exectute runs
class RunContext(cogment.Context):
    def __init__(
        self,
        user_id,
        cog_settings,
        services_endpoints,
        asyncio_loop=None,
        prometheus_registry=REGISTRY,
    ):
        super().__init__(
            user_id,
            cog_settings,
            asyncio_loop=asyncio_loop,
            prometheus_registry=prometheus_registry,
        )

        self._cog_settings = cog_settings
        self._services_endpoints = services_endpoints

        # Pre trial hook => actor/environment config + services urls resolution
        async def pre_trial_hook(pre_trial_hook_session):
            log.debug(f"[pre_trial_hook] Configuring trial {pre_trial_hook_session.get_trial_id()}")

            environment_params = pre_trial_hook_session.trial_config.environment
            pre_trial_hook_session.environment_config = environment_params.config
            pre_trial_hook_session.environment_implementation = environment_params.specs.implementation
            pre_trial_hook_session.environment_endpoint = "grpc://" + self._get_service_endpoint(
                pre_trial_hook_session.environment_implementation
            )
            pre_trial_hook_session.datalog_endpoint = "grpc://" + services_endpoints["trial_datastore"]
            pre_trial_hook_session.actors = [
                {
                    "name": actor.name,
                    "actor_class": actor.actor_class,
                    "endpoint": "client"
                    if actor.implementation == "client"
                    else ("grpc://" + self._get_service_endpoint(actor.implementation)),
                    "implementation": "" if actor.implementation == "client" else actor.implementation,
                    "config": set_config_index(getattr(actor, actor.WhichOneof("config_oneof")), actor_idx),
                }
                for actor_idx, actor in enumerate(pre_trial_hook_session.trial_config.actors)
            ]

            pre_trial_hook_session.validate()

        self.register_pre_trial_hook(pre_trial_hook)

        self._run_impls = {}

        # Cache used by the model registry
        self._model_registry_cache = LRU()

    def _get_service_endpoint(self, services_name):
        if services_name not in self._services_endpoints:
            raise Exception(f"unknown service [{services_name}]")

        desired_service_endpoints = self._services_endpoints[services_name]

        if not desired_service_endpoints:
            raise Exception(f"no endpoint defined for service [{services_name}]")

        if isinstance(desired_service_endpoints, list):
            return random.choice(desired_service_endpoints)
        return desired_service_endpoints

    def register_run(self, run_impl, run_sample_producer_impl, impl_name, default_config):
        if self._grpc_server is not None:
            raise RuntimeError("Cannot register a run after the server is started")
        if impl_name in self._run_impls:
            raise RuntimeError(f"The run implementation name must be unique: [{impl_name}]")
        self._run_impls[impl_name] = (
            run_impl,
            run_sample_producer_impl,
            default_config,
        )

    def _get_controller(self):
        return self.get_controller(endpoint=cogment.Endpoint(self._get_service_endpoint("orchestrator")))

    def _get_trial_datastore_client(self):
        return TrialDatastoreClient(endpoint=self._get_service_endpoint("trial_datastore"))

    def get_model_registry_client(self):
        return ModelRegistryClient(
            endpoint=self._get_service_endpoint("model_registry"),
            cache=self._model_registry_cache,
        )

    def _create_run_session(self, run_params_name, run_implementation, serialized_config, run_id=None):
        if run_implementation not in self._run_impls:
            raise RuntimeError(f"Unknown run implementation [{run_implementation}]")

        (run_impl, run_sample_producer_impl, default_config) = self._run_impls[run_implementation]

        merged_config = copy.deepcopy(default_config)
        if serialized_config is not None:
            merged_config.MergeFromString(serialized_config)

        return RunSession(
            cog_settings=self._cog_settings,
            controller=self._get_controller(),
            trial_datastore_client=self._get_trial_datastore_client(),
            config=merged_config,
            run_sample_producer_impl=run_sample_producer_impl,
            impl_name=run_implementation,
            run_impl=run_impl,
            params_name=run_params_name,
            run_id=run_id,
        )

    async def exec_run(self, impl_name, config=None, run_id=None):
        if impl_name not in self._run_impls:
            raise RuntimeError(f"Unknown run implementation [{impl_name}]")

        (run_impl, run_sample_producer_impl, default_config) = self._run_impls[impl_name]

        merged_config = copy.deepcopy(default_config)
        if config is not None:
            merged_config.MergeFrom(config)

        run_session = RunSession(
            cog_settings=self._cog_settings,
            controller=self._get_controller(),
            trial_datastore_client=self._get_trial_datastore_client(),
            config=merged_config,
            run_sample_producer_impl=run_sample_producer_impl,
            impl_name=impl_name,
            run_impl=run_impl,
            params_name="manual_run",
            run_id=run_id,
        )

        await run_session.exec()

    async def serve_all_registered(
        self,
        served_endpoint,
        prometheus_port=cogment.context.DEFAULT_PROMETHEUS_PORT,
    ):
        serve_all_registered_task = asyncio.create_task(super().serve_all_registered(served_endpoint, prometheus_port))

        while self._grpc_server is None:
            await asyncio.sleep(0.1)

        if self._run_impls:
            servicer = RunServicer(self._create_run_session)
            add_RunServicer_to_server(servicer, self._grpc_server)

        reflection.enable_server_reflection(
            (
                RUN_DESCRIPTOR.services_by_name["Run"].full_name,
                reflection.SERVICE_NAME,
            ),
            self._grpc_server,
        )

        try:
            await serve_all_registered_task
        except Exception as error:
            log.info(f"Properly canceling the server task after {error} ...")
            serve_all_registered_task.cancel()
            await serve_all_registered_task
            raise error
