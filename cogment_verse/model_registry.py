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

import abc
import asyncio
import copy
import io
import logging
import math
import time
from urllib.parse import urlparse

import grpc.aio
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
from cogment.model_registry import (
    GRPC_BYTE_SIZE_LIMIT,
    MODEL_REGISTRY_RETRIEVE_VERSION_TIME,
    MODEL_REGISTRY_STORE_VERSION_TIME,
)
from google.protobuf.json_format import MessageToDict

from cogment_verse.services_directory import ServiceType
from cogment_verse.utils import LRU

log = logging.getLogger(__name__)


class Model(abc.ABC):
    def __init__(self, model_id, version_number=0):
        self.model_id = model_id
        self.version_number = version_number

    @abc.abstractmethod
    def get_model_user_data(self):
        """
        Retrieve the user data associated with the model instance
        Returns:
            model_user_data (dict[str, str]): model user data that will be saved alongside the model
        """
        return {}

    @classmethod
    @abc.abstractmethod
    def load(cls, model_id, version_number, model_user_data, version_user_data, model_data_f):
        """
        Load a serialized model instance and return it
        Args:
            model_id (string): unique identifier for the model
            model_user_data (dict[str, str]): model user data
            version_user_data (dict[str, str]): version user data
            model_data_f: file object that will be used to load the version model data
        Returns:
            model: the loaded model
        """

    @abc.abstractmethod
    def save(self, model_data_f):
        """
        Serialize and save the model
        Args:
            model_data_f: file object that will be used to save the version model data
        Returns:
            version_user_data (dict[str, str]): version user data that will be saved alongside the model version
        """


class ModelRegistry:
    def __init__(self, services_directory):
        self._service_directory = services_directory
        self._cache = LRU()

    def _get_grpc_stub(self):
        endpoint = self._service_directory.get(ServiceType.MODEL_REGISTRY)
        endpoint_components = urlparse(endpoint)
        if endpoint_components.scheme != "grpc":
            raise RuntimeError(f"Unsupported scheme for [{endpoint}], expected 'grpc'")
        channel = grpc.aio.insecure_channel(endpoint_components.netloc)
        return ModelRegistrySPStub(channel)

    @staticmethod
    def _build_model_version_data_cache_key(data_hash):
        return f"model_version_data_{data_hash}"

    @staticmethod
    def _build_model_info_cache_key(model_id):
        return f"model_info_{model_id}"

    async def publish_initial_version(self, model):
        """
        Create a new model in the model registry and publish the initial version

        Parameters:
            model (Model): The model
        Returns:
            model_info, version_info (dict, dict): A tuple containing the information of the model and the information of the published initial version
        """
        model_user_data_str = {}
        for key, value in model.get_model_user_data().items():
            model_user_data_str[key] = str(value)

        model_info = ModelInfo(model_id=model.model_id, user_data=model_user_data_str)

        req = CreateOrUpdateModelRequest(model_info=model_info)
        await self._get_grpc_stub().CreateOrUpdateModel(req)

        self._cache[self._build_model_info_cache_key(model.model_id)] = model_info

        version_info = await self.publish_version(model)

        return (model_info, version_info)

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
            rep = await self._get_grpc_stub().RetrieveModels(req)

            model_info = rep.model_infos[0]

            self._cache[cache_key] = model_info

        return MessageToDict(self._cache[cache_key], preserving_proto_field_name=True)

    async def publish_version(self, model, archived=False):
        """
        Publish a new version of the model

        Parameters:
            model (Model): The model
            archive (bool - default is False): If true, the model version will be archived (i.e. stored in permanent storage)
        Returns
            version_info (dict): The information of the published version
        """

        def generate_chunks():
            try:
                with io.BytesIO() as model_data_io:
                    version_user_data = model.save(model_data_io)
                    version_data = model_data_io.getvalue()

                version_info = ModelVersionInfo(model_id=model.model_id, archived=archived, data_size=len(version_data))
                for key, value in version_user_data.items():
                    version_info.user_data[key] = str(value)  # pylint: disable=no-member

                yield CreateVersionRequestChunk(header=CreateVersionRequestChunk.Header(version_info=version_info))

                chunksize = math.trunc(GRPC_BYTE_SIZE_LIMIT / 2)  # 2MB to keep under well under the GRPC 4MB limit
                sent_chunk_num = 0
                while version_data:
                    yield CreateVersionRequestChunk(
                        body=CreateVersionRequestChunk.Body(data_chunk=version_data[:chunksize])
                    )
                    version_data = version_data[chunksize:]
                    sent_chunk_num += 1

            except Exception as error:
                log.error("Error while generating model version chunk", exc_info=error)
                raise error

        with MODEL_REGISTRY_STORE_VERSION_TIME.labels(model_id=model.model_id).time():
            rep = await self._get_grpc_stub().CreateVersion(generate_chunks())

        cache_key = self._build_model_version_data_cache_key(rep.version_info.data_hash)
        # Store a copy in the local cache
        self._cache[cache_key] = copy.deepcopy(model)

        return MessageToDict(rep.version_info, preserving_proto_field_name=True)

    async def retrieve_version(self, model_cls, model_id, version_number=-1):
        """
        Retrieve a version of the model

        Parameters:
            model_cls (Model class): The class of the model
            model_id (string): Unique id of the model
            version_number (int - default is -1): The version number (-1 for the latest)
        Returns
            model, model_info, version_info (ModelT, dict[str, str], dict[str, str]): A tuple containing the model, the model info and the version info
        """
        start_time = time.time()

        # First retrieve the model info and model version info
        async def retrieve_version_info(model_id, version_number):
            req = RetrieveVersionInfosRequest(model_id=model_id, version_numbers=[version_number])
            rep = await self._get_grpc_stub().RetrieveVersionInfos(req)
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
            async for chunk in self._get_grpc_stub().RetrieveVersionData(req):
                data += chunk.data_chunk

            model = model_cls.load(
                model_id, version_number, model_info["user_data"], version_info["user_data"], io.BytesIO(data)
            )
            assert model.model_id == model_id
            self._cache[cache_key] = model

        MODEL_REGISTRY_RETRIEVE_VERSION_TIME.labels(model_id=model_id, cached=cached).observe(time.time() - start_time)

        return self._cache[cache_key], model_info, version_info
