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
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

import cogment.api.model_registry_pb2 as model_registry_api
import grpc.aio
from cogment.api.model_registry_pb2 import ModelInfo
from cogment.api.model_registry_pb2_grpc import ModelRegistrySPStub
from cogment.model_registry import (GRPC_BYTE_SIZE_LIMIT, MODEL_REGISTRY_RETRIEVE_VERSION_TIME,
                                    MODEL_REGISTRY_STORE_VERSION_TIME)

from cogment_verse.services_directory import ServiceType
from cogment_verse.utils import LRU

log = logging.getLogger(__name__)


class ModelInfo:
    def __init__(self, model_id: str, model_user_data: Dict[str, str]):
        self.id = model_id
        self.user_data = model_user_data

    def __str__(self):
        return f"ModelInfo(id={self.id}, user_data={self.user_data})"


class VersionInfo:
    def __init__(self, proto_version_info):
        self.version_number = proto_version_info.version_number
        self.creation_timestamp = proto_version_info.creation_timestamp
        self.archived = proto_version_info.archived
        self.data_hash = proto_version_info.data_hash
        self.data_size = proto_version_info.data_size

    def __str__(self):
        return f"VersionInfo(version_number={self.version_number}, creation_timestamp={self.creation_timestamp}, archived={self.archived}, data_size={self.data_size})"


class Model(ModelInfo):
    def __init__(self, model_id, version_number=0):
        super().__init__(model_id, {})
        self.version_user_data: Dict[str, str] = {}
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
        self._info_cache: Dict[str, ModelInfo] = LRU()
        self._data_cache: Dict[str, Model] = LRU()

    def _get_grpc_stub(self):
        endpoint = self._service_directory.get(ServiceType.MODEL_REGISTRY)
        endpoint_components = urlparse(endpoint)
        if endpoint_components.scheme != "grpc":
            raise RuntimeError(f"Unsupported scheme for [{endpoint}], expected 'grpc'")
        channel = grpc.aio.insecure_channel(endpoint_components.netloc)
        return ModelRegistrySPStub(channel)

    async def store_initial_version(self, model: Model) -> VersionInfo:
        """
        Create a new model in the model registry and store the initial version

        Parameters:
            model (Model): The model
        Returns:
            version_info (VersionInfo):
            The information of the stored initial version
        """
        model_user_data_str = {}
        for key, value in model.get_model_user_data().items():
            model_user_data_str[key] = str(value)

        registry_model_info = model_registry_api.ModelInfo(model_id=model.id, user_data=model_user_data_str)
        cached_model_info = ModelInfo(model.id, model_user_data_str)

        req = model_registry_api.CreateOrUpdateModelRequest(model_info=registry_model_info)
        await self._get_grpc_stub().CreateOrUpdateModel(req)

        self._info_cache[model.id] = cached_model_info
        version_info = await self.store_version(model)
        return version_info

    async def retrieve_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Retrieve the given's model information

        Parameters:
            model_id (string): The model id
        Returns
            model_info (ModelInfo): The information of the model
        """
        if model_id not in self._info_cache:
            req = model_registry_api.RetrieveModelsRequest(model_ids=[model_id])
            try:
                rep = await self._get_grpc_stub().RetrieveModels(req)
            except Exception:
                log.error(f"Error retrieving model version with id [{model_id}]")
                return None

            registry_model_info = rep.model_infos[0]
            cached_model_info = ModelInfo(registry_model_info.model_id, registry_model_info.user_data)

            self._info_cache[model_id] = cached_model_info

        return self._info_cache[model_id]

    async def store_version(self, model: Model, archived=False) -> VersionInfo:
        """
        Store a new version of the model

        Parameters:
            model (Model): The model
            archive (bool - default is False):
            If true, the model version will be archived (i.e. stored in permanent storage)
        Returns
            version_info (VersionInfo): The information of the stored version
        """

        def generate_chunks():
            try:
                with io.BytesIO() as model_data_io:
                    version_user_data = model.save(model_data_io)
                    version_data = model_data_io.getvalue()

                version_info = model_registry_api.ModelVersionInfo(model_id=model.id, archived=archived, data_size=len(version_data))
                for key, value in version_user_data.items():
                    version_info.user_data[key] = str(value)  # pylint: disable=no-member

                chunk_header = model_registry_api.CreateVersionRequestChunk.Header(version_info=version_info)
                yield model_registry_api.CreateVersionRequestChunk(header=chunk_header)

                chunksize = math.trunc(GRPC_BYTE_SIZE_LIMIT / 2)

                chunked_version_data = [
                    version_data[index:index + chunksize] for index in range(0, len(version_data), chunksize)
                ]
                for data_chunk in chunked_version_data:
                    chunk_body = model_registry_api.CreateVersionRequestChunk.Body(data_chunk=data_chunk)
                    yield model_registry_api.CreateVersionRequestChunk(body=chunk_body)

            except Exception as error:
                log.error("Error while generating model version chunk", exc_info=error)
                raise error

        with MODEL_REGISTRY_STORE_VERSION_TIME.labels(model_id=model.id).time():
            rep = await self._get_grpc_stub().CreateVersion(generate_chunks())

        self._data_cache[rep.version_info.data_hash] = copy.deepcopy(model)

        return VersionInfo(rep.version_info)

    async def retrieve_version(self, model_cls, model_id: str, version_number=-1) -> Optional[Model]:
        """
        Retrieve a version of the model

        Parameters:
            model_cls (Model class): The class of the model
            model_id (string): Unique id of the model
            version_number (int - default is -1): The version number (-1 for the latest)
        Returns
            model (ModelT): The stored model.
        """
        start_time = time.time()

        # First retrieve the model info and model version info
        async def retrieve_version_info(model_id, version_number):
            req = model_registry_api.RetrieveVersionInfosRequest(model_id=model_id, version_numbers=[version_number])
            try:
                rep = await self._get_grpc_stub().RetrieveVersionInfos(req)
            except Exception:
                log.error(
                    f"Failed to retrieve model version with id [{model_id}] and version number [{version_number}]"
                )
                return None

            version_info_pb = rep.version_infos[0]
            return version_info_pb

        [model_info, version_info] = await asyncio.gather(
            self.retrieve_model_info(model_id), retrieve_version_info(model_id, version_number)
        )

        if model_info is None or version_info is None:
            return None

        cached = version_info.data_hash in self._data_cache

        # Check if the model version data is already in memory, if not retrieve
        if cached:
            model = self._data_cache[version_info.data_hash]
        if not cached:
            req = model_registry_api.RetrieveVersionDataRequest(
                model_id=model_id, version_number=version_info.version_number)
            data = b""
            async for chunk in self._get_grpc_stub().RetrieveVersionData(req):
                data += chunk.data_chunk

            model = model_cls.load(
                model_id, version_number, model_info.user_data, version_info.user_data, io.BytesIO(data)
            )
            assert model.id == model_id
            self._data_cache[version_info.data_hash] = model

        MODEL_REGISTRY_RETRIEVE_VERSION_TIME.labels(model_id=model_id, cached=cached).observe(time.time() - start_time)

        return model
