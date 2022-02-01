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

import grpc.aio
from cogment.api.trial_datastore_pb2 import RetrieveSamplesRequest, RetrieveTrialsRequest
from cogment.api.trial_datastore_pb2_grpc import TrialDatastoreSPStub


class TrialDatastoreClient:
    def __init__(self, endpoint):
        channel = grpc.aio.insecure_channel(endpoint)
        self._stub = TrialDatastoreSPStub(channel)

    async def retrieve_trials(self, trial_ids, timeout=30000):
        req = RetrieveTrialsRequest(trial_ids=trial_ids, timeout=timeout)

        rep = await self._stub.RetrieveTrials(req)

        return rep.trial_infos

    async def retrieve_samples(self, trial_ids):
        req = RetrieveSamplesRequest(trial_ids=trial_ids)
        rep_stream = self._stub.RetrieveSamples(req)

        async def sample_generator():
            async for rep_msg in rep_stream:
                yield rep_msg.trial_sample

        return sample_generator
