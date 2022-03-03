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
import io
import logging
import time

import grpc.aio
from google.protobuf.json_format import MessageToDict
from cogment.api.model_registry_pb2 import (
    CreateOrUpdateModelRequest,
    CreateVersionRequestChunk,
    ModelInfo,
    ModelVersionInfo,
    RetrieveModelsRequest,
    RetrieveVersionDataRequest,
    RetrieveVersionInfosRequest,
)
from cogment.api.model_registry_pb2_grpc import ModelRegistrySPStub
from cogment_verse.utils import LRU
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
    def __init__(self, endpoint, cache=LRU()):

        channel = grpc.aio.insecure_channel(endpoint)
        self._stub = ModelRegistrySPStub(channel)

        self._cache = cache

    @staticmethod
    def _build_model_version_data_cache_key(data_hash):
        return f"model_version_data_{data_hash}"

    @staticmethod
    def _build_model_info_cache_key(model_id):
        return f"model_info_{model_id}"

    async def create_model(self, model_id, model_user_data=None):
        """
        Create a new model in the model registry

        Parameters:
            model_id (string): The model id
            model_user_data (dict[str, str] - optional): model user data
        """
        model_user_data_str = {}
        if model_user_data:
            for key, value in model_user_data.items():
                model_user_data_str[key] = str(value)

        model_info = ModelInfo(model_id=model_id, user_data=model_user_data_str)

        req = CreateOrUpdateModelRequest(model_info=model_info)
        await self._stub.CreateOrUpdateModel(req)

        self._cache[self._build_model_info_cache_key(model_id)] = model_info

    async def retrieve_model_info(self, model_id):
        """
        Retrieve the given's model information

        Parameters:
            model_id (string): The model id
        Returns
            model_info (dict): The information of the model
        """
        cache_key = self._build_model_info_cache_key(model_id)

        if cache_key not in self._cache:
            req = RetrieveModelsRequest(model_ids=[model_id])
            rep = await self._stub.RetrieveModels(req)

            model_info = rep.model_infos[0]

            self._cache[cache_key] = model_info

        return MessageToDict(self._cache[cache_key], preserving_proto_field_name=True)

    async def publish_version(self, model_id, model, save_model, archived=False, **kwargs):
        """
        Publish a new version of the model

        Parameters:
            model_id (string): Unique id of the model
            model (ModelT): The model
            save_model (f(ModelT, dict[str, str], BinaryIO, **kwargs) -> dict[str, str]): A function able to save the model, returning version_user_data
            archive (bool - default is False): If true, the model version will be archived (i.e. stored in permanent storage)
            kwargs: any number of key/values parameters, forwarded to `save_model`
        Returns
            version_info (dict): The information of the published version
        """

        model_info = await self.retrieve_model_info(model_id)
        model_user_data = model_info["user_data"]

        def generate_chunks():
            try:
                with io.BytesIO() as model_data_io:
                    version_user_data = save_model(model, model_user_data, model_data_io, **kwargs)
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

        cache_key = self._build_model_version_data_cache_key(rep.version_info.data_hash)
        self._cache[cache_key] = model

        return MessageToDict(rep.version_info, preserving_proto_field_name=True)

    async def retrieve_version(self, model_id, load_model, version_number=-1, **kwargs):
        """
        Retrieve a version of the model

        Parameters:
            model_id (string): Unique id of the model
            load_model (f(string, int, dict[str, str], dict[str, str], BinaryIO)): A function able to load the model
            version_number (int - default is -1): The version number (-1 for the latest)
            kwargs: any number of key/values parameters, forwarded to `load_model`
        Returns
            model, model_info, version_info (ModelT, dict[str, str], dict[str, str]): A tuple containing the model version data, the model info and the model version info
        """
        start_time = time.time()

        # First retrieve the model info and model version info
        async def retrieve_version_info(model_id, version_number):
            req = RetrieveVersionInfosRequest(model_id=model_id, version_numbers=[version_number])
            rep = await self._stub.RetrieveVersionInfos(req)
            version_info_pb = rep.version_infos[0]
            version_info = MessageToDict(version_info_pb, preserving_proto_field_name=True)
            return version_info

        [model_info, version_info] = await asyncio.gather(
            self.retrieve_model_info(model_id), retrieve_version_info(model_id, version_number)
        )

        cache_key = self._build_model_version_data_cache_key(version_info["data_hash"])
        cached = cache_key in self._cache

        # Check if the model version data is already in memory, if not retrieve
        if not cached:
            req = RetrieveVersionDataRequest(model_id=model_id, version_number=version_info["version_number"])
            data = b""
            async for chunk in self._stub.RetrieveVersionData(req):
                data += chunk.data_chunk

            model = load_model(
                model_id, version_number, model_info["user_data"], version_info["user_data"], io.BytesIO(data), **kwargs
            )
            self._cache[cache_key] = model

        MODEL_REGISTRY_RETRIEVE_VERSION_TIME.labels(model_id=model_id, cached=cached).observe(time.time() - start_time)

        return self._cache[cache_key], model_info, version_info
