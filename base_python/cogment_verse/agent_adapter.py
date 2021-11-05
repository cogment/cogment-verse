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

from cogment_verse.model_registry_client import get_model_registry_client

from cogment_verse.utils import LRU

import abc
import logging

log = logging.getLogger(__name__)


class AgentAdapter(abc.ABC):
    def __init__(self):
        """
        Create an agent adapter
        """
        self._model_cache = LRU()

    @abc.abstractmethod
    def _create(self, model_id, **kwargs):
        """
        Create and return a model instance
        Parameters:
            model_id (string): unique identifier for the model
            kwargs: any number of key/values paramters
        Returns:
            model: the created model
        """

    @abc.abstractmethod
    def _load(self, model_id, version_number, version_user_data, model_data_f):
        """
        Load a serialized model instance and return it
        Args:
            model_id (string): unique identifier for the model
            version_number (int): version number of the data
            version_user_data (dict[str, str]): version user data
            model_data_f: file object that will be used to load the version model data
        Returns:
            model: the loaded model
        """

    @abc.abstractmethod
    def _save(self, model, model_data_f):
        """
        Serialize and save a model
        Args:
            model: a model, as returned by method of this class
            model_data_f: file object that will be used to save the version model data
        Returns:
            version_user_data (dict[str, str]): version user data
        """

    async def create_and_publish_initial_version(self, model_id, **kwargs):
        """
        Create and publish to the model registry a model instance
        Parameters:
            model_id (string): unique identifier for the model
            kwargs: any number of key/values paramters
        Returns:
            model, version_info: the created model and information for the initial published version
        """

        # Create the agent model locally
        model = self._create(model_id, **kwargs)

        # Create it in the model registry
        await get_model_registry_client().create_model(model_id)

        # Publish the first version
        version_info = await self.publish_version(model_id, model)
        return model, version_info

    async def publish_version(self, model_id, model, archived=False):
        """
        Publish to the model registry a new version of a model
        Parameters:
            model_id (string): unique identifier for the model
            model: a model, as returned by method of this class
            archive (bool - default is False): If true, the model version will be archived (i.e. stored in permanent storage)
        Returns:
            version_info: information for the initial published version
        """
        return await get_model_registry_client().publish_model_version(
            model_id, model, self._save, cache=self._model_cache, archived=archived
        )

    async def retrieve_version(self, model_id, version_number=-1):
        """
        Publish to the model registry a new version of a model
        Parameters:
            model_id (string): Unique id of the model
            version_number (int - default is -1): The version number (-1 for the latest)
        Returns:
            model, version_info: the retrieve model and information for the retrieved version
        """
        return await get_model_registry_client().retrieve_model_version(
            model_id, self._load, self._model_cache, version_number=version_number
        )

    @abc.abstractmethod
    def _create_actor_implementations(self):
        """
        Create all the available actor implementation for this adapter
        Returns:
            dict[impl_name: string, (actor_impl: Callable, actor_classes: []string)]: key/value definition for the available actor implementations.
        """

    @abc.abstractmethod
    def _create_run_implementations(self):
        """
        Create all the available run implementation for this adapter
        Returns:
            dict[impl_name: string, (sample_producer_impl: Callable, run_impl: Callable, default_run_config)]: key/value definition for the available run implementations.
        """

    def register_implementations(self, context):
        """
        Register all the implementations defined in this adapter
        Parameters:
            context: Cogment context with which the implementations are adapted
        """
        for impl_name, (actor_impl, actor_classes) in self._create_actor_implementations().items():
            log.info(f"Registering actor implementation [{impl_name}]")
            context.register_actor(impl=actor_impl, impl_name=impl_name, actor_classes=actor_classes)

        for impl_name, (sample_producer_impl, run_impl, default_config) in self._create_run_implementations().items():
            log.info(f"Registering run implementation [{impl_name}]")
            context.register_run(
                run_impl=run_impl,
                run_sample_producer_impl=sample_producer_impl,
                impl_name=impl_name,
                default_config=default_config,
            )
