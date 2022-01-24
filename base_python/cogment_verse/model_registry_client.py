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

import io
import logging
import os
import time

import grpc.aio
from cogment.api.model_registry_pb2 import (
    CreateOrUpdateModelRequest,
    CreateVersionRequestChunk,
    ModelInfo,
    ModelVersionInfo,
    RetrieveVersionDataRequest,
    RetrieveVersionInfosRequest,
)
from cogment.api.model_registry_pb2_grpc import ModelRegistrySPStub
from google.protobuf.json_format import MessageToDict
from prometheus_client import Summary

MODEL_REGISTRY_PUBLISH_VERSION_TIME = Summary(
    "model_registry_publish_version_seconds",
    "Time spent serializing and sending the model to the registry",
    ["model_id"],
)
MODEL_REGISTRY_RETRIEVE_VERSION_TIME = Summary(
    "model_registry_retrieve_version_seconds",
    "Time spent retrieving and deserializing the agent model version from the registry",
    ["model_id", "cached"],
)

log = logging.getLogger(__name__)


class ModelRegistryClient:
    def __init__(self, endpoint):

        channel = grpc.aio.insecure_channel(endpoint)
        self._stub = ModelRegistrySPStub(channel)

    async def create_model(self, model_id):
        """
        Create a new model in the model registry

        Parameters:
            model_id: The model id
        """

        req = CreateOrUpdateModelRequest(model_info=ModelInfo(model_id=model_id))
        await self._stub.CreateOrUpdateModel(req)

    async def publish_model_version(self, model_id, model, save_model, cache, archived=False):
        """
        Publish a new version of the model

        Parameters:
            model_id (string): Unique id of the model
            model (ModelT): The model
            save_model (f(ModelT, BinaryIO)): A function able to save the model
            cache: A dict-like structure used to save the model version locally addressed by its hash
            archive (bool - default is False): If true, the model version will be archived (i.e. stored in permanent storage)
        Returns
            version_info (dict): The information of the published version
        """

        def generate_chunks():
            try:
                with io.BytesIO() as model_data_io:
                    version_user_data = save_model(model, model_data_io)
                    version_data = model_data_io.getvalue()

                version_info = ModelVersionInfo(model_id=model_id, archived=archived, data_size=len(version_data))
                for key, value in version_user_data.items():
                    version_info.user_data[key] = str(value)

                yield CreateVersionRequestChunk(header=CreateVersionRequestChunk.Header(version_info=version_info))

                chunksize = 2 * 1024 * 1024  # 2MB to keep under well under the GRPC 4MB
                while version_data:
                    yield CreateVersionRequestChunk(
                        body=CreateVersionRequestChunk.Body(data_chunk=version_data[:chunksize])
                    )
                    version_data = version_data[chunksize:]
            except Exception as error:
                log.error("Error while generating model version chunk", exc_info=error)
                raise error

        with MODEL_REGISTRY_PUBLISH_VERSION_TIME.labels(model_id=model_id).time():
            rep = await self._stub.CreateVersion(generate_chunks())

        cache[rep.version_info.data_hash] = model

        return MessageToDict(rep.version_info, preserving_proto_field_name=True)

    async def retrieve_model_version(self, model_id, load_model, cache, version_number=-1):
        """
        Retrieve a version of the model

        Parameters:
            model_id (string): Unique id of the model
            load_model (f(string, int, dict[str, str], BinaryIO)): A function able to load the model
            cache: A dict-like structure used to retrieve the model version locally addressed by its hash
            version_number (int - default is -1): The version number (-1 for the latest)
        Returns
            model: The model at the retrieved version
        """
        start_time = time.time()

        # First retrieve the model version info
        req = RetrieveVersionInfosRequest(model_id=model_id, version_numbers=[version_number])
        rep = await self._stub.RetrieveVersionInfos(req)
        version_info = rep.version_infos[0]

        cached = version_info.data_hash in cache

        # Check if the model is already in memory, if not retrieve
        if not cached:
            req = RetrieveVersionDataRequest(model_id=model_id, version_number=version_info.version_number)
            data = b""
            async for chunk in self._stub.RetrieveVersionData(req):
                data += chunk.data_chunk

            version_user_data = {}
            for key, value in version_info.user_data.items():
                version_user_data[key] = value

            model = load_model(
                model_id,
                version_number,
                version_user_data,
                io.BytesIO(data),
            )
            cache[version_info.data_hash] = model

        MODEL_REGISTRY_RETRIEVE_VERSION_TIME.labels(model_id=model_id, cached=cached).observe(time.time() - start_time)

        return cache[version_info.data_hash], MessageToDict(version_info, preserving_proto_field_name=True)


def get_model_registry_client():
    return ModelRegistryClient(endpoint=os.getenv("COGMENT_VERSE_MODEL_REGISTRY_ENDPOINT"))
