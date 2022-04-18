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

from cogment_verse.api.run_api_pb2 import ListRunsReply, RunInfo, RunStatus, StartRunReply, TerminateRunReply
from cogment_verse.api.run_api_pb2_grpc import RunServicer as AbstractRunServicer
from cogment_verse.run.run_session import RunSessionStatus
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp

# pylint: disable=invalid-overridden-method, invalid-name, no-member

log = logging.getLogger(__name__)


def run_info_from_run_session(run_session):
    run_session_status = run_session.get_status()
    status = RunStatus.UNKNOWN
    if run_session_status is RunSessionStatus.RUNNING:
        status = RunStatus.RUNNING
    elif run_session_status is RunSessionStatus.SUCCESS:
        status = RunStatus.SUCCESS
    elif run_session_status is RunSessionStatus.TERMINATED:
        status = RunStatus.TERMINATED
    elif run_session_status is RunSessionStatus.ERROR:
        status = RunStatus.ERROR

    start_timestamp = Timestamp()
    start_timestamp.FromDatetime(run_session.start_time)

    return RunInfo(
        run_id=run_session.run_id,
        params_name=run_session.params_name,
        implementation=run_session.impl_name,
        start_timestamp=start_timestamp,
        status=status,
        steps_count=run_session.count_steps(),
    )


class RunServicer(AbstractRunServicer):
    def __init__(self, create_run_session):
        self._create_run_session = create_run_session
        self._run_sessions = {}

    async def StartRun(self, request, _context):
        run_id = request.run_id if request.run_id != "" else None
        log.debug(f"StartRun(run_id=[{run_id}], run_params=[{MessageToDict(request.run_params)}])")
        if run_id is not None and run_id in self._run_sessions:
            raise RuntimeError(f"Run [{run_id}] already served")

        run_session = self._create_run_session(
            run_params_name=request.run_params.name,
            run_implementation=request.run_params.implementation,
            serialized_config=request.run_params.config.content,
            run_id=run_id,
        )
        run_session.exec()
        self._run_sessions[run_session.run_id] = run_session

        try:
            return StartRunReply(run=run_info_from_run_session(run_session))
        except:
            await run_session.terminate()
            raise

    def ListRuns(self, _request, _context):
        log.debug("ListRuns()")

        runs_reply = ListRunsReply()

        for run_info in sorted(
            [run_info_from_run_session(s) for s in self._run_sessions.values()],
            key=lambda info: info.start_timestamp.ToNanoseconds(),
        ):
            runs_reply.runs.append(run_info)

        return runs_reply

    async def TerminateRun(self, request, _context):
        run_id = request.run_id
        log.debug(f"TerminateRun(run_id=[{run_id}])")
        if run_id not in self._run_sessions:
            raise RuntimeError(f"Unknown run [{run_id}]")

        run_session = self._run_sessions[run_id]
        await run_session.terminate()

        return TerminateRunReply(run=run_info_from_run_session(run_session))
