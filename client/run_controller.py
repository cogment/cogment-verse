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

import logging
from urllib.parse import urlparse

import grpc.aio
from google.protobuf.json_format import MessageToDict
from run_api_pb2 import ListRunsRequest, RunParams, RunStatus, StartRunRequest, TerminateRunRequest
from run_api_pb2_grpc import RunStub

log = logging.getLogger(__name__)

# pylint: disable=no-member
class RunController:
    def __init__(self, endpoints):
        self._endpoints = []
        for endpoint_url in endpoints:
            endpoint_components = urlparse(endpoint_url)
            if endpoint_components.scheme != "grpc":
                raise RuntimeError(f"Unsupported scheme for [{endpoint_url}], expected 'grpc'")
            self._endpoints.append((endpoint_url, RunStub(grpc.aio.insecure_channel(endpoint_components.netloc))))

    async def start_run(self, name, implementation, config, run_id=None):
        log.debug(
            f"start_run(run_id=[{run_id}], name=[{name}], implementation=[{implementation}], config=[{MessageToDict(config)}])"
        )
        min_running_run_count = 100000
        for current_endpoint, current_stub in self._endpoints:
            req = ListRunsRequest()
            rep = await current_stub.ListRuns(req)

            running_run_count = len([run.status == RunStatus.RUNNING for run in rep.runs])
            if running_run_count < min_running_run_count:
                min_running_run_count = running_run_count
                endpoint, stub = current_endpoint, current_stub

        run_params = RunParams(name=name, implementation=implementation)
        if config is not None:
            run_params.config.content = config.SerializeToString()

        req = StartRunRequest(run_params=run_params, run_id=run_id)
        rep = await stub.StartRun(req)
        return {
            **MessageToDict(rep.run),
            "endpoint": endpoint,
        }

    async def list_runs(self):
        log.debug("list_runs()")
        runs = []
        for endpoint, stub in self._endpoints:
            req = ListRunsRequest()
            rep = await stub.ListRuns(req)
            runs.extend([{**MessageToDict(run), "endpoint": endpoint} for run in rep.runs])
        return runs

    async def terminate_run(self, run_id):
        log.debug(f"terminate_run(run_id=[{run_id}])")
        for endpoint, stub in self._endpoints:
            req = ListRunsRequest()
            rep = await stub.ListRuns(req)

            has_target_run = any(run.run_id == run_id for run in rep.runs)
            if has_target_run:
                req = TerminateRunRequest(run_id=run_id)
                rep = await stub.TerminateRun(req)
                return {
                    **MessageToDict(rep.run),
                    "endpoint": endpoint,
                }
        raise RuntimeError(f"Unknown run [{run_id}]")
