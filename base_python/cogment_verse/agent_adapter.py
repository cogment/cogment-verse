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

import abc
import logging

from cogment_verse.utils import LRU, get_full_class_name

log = logging.getLogger(__name__)


class AgentAdapter(abc.ABC):
    MODEL_USER_DATA_ADAPTER_CLASS_NAME_KEY = "_source_adapter_class_name"
    VERSION_USER_DATA_MODEL_CLASS_NAME_KEY = "_model_class_name"

    def __init__(self):
        """
        Create an agent adapter
        """
        self._model_cache = LRU()
        self._adapter_class_name = get_full_class_name(self)

        def default_get_model_registry_client():
            raise RuntimeError("`get_model_registry_client` is not defined before a call to `register_implementations`")

        self.get_model_registry_client = default_get_model_registry_client

    def _create(self, model_id, **kwargs):
        """
        Create and return a model instance
        Parameters:
            model_id (string): unique identifier for the model
            kwargs: any number of key/values paramters, forwarded from `create_and_publish_initial_version`
        Returns:
            model, model_user_data: a tuple containing the created model and additional user_data
        """
        raise NotImplementedError

    def __create(self, model_id, **kwargs):
        model, model_user_data = self._create(model_id, **kwargs)
        model_user_data[self.MODEL_USER_DATA_ADAPTER_CLASS_NAME_KEY] = self._adapter_class_name
        return model, model_user_data

    def _load(self, model_id, version_number, model_user_data, version_user_data, model_data_f, **kwargs):
        """
        Load a serialized model instance and return it
        Args:
            model_id (string): unique identifier for the model
            version_number (int): version number of the data
            model_user_data (dict[str, str]): model user data
            version_user_data (dict[str, str]): version user data
            model_data_f: file object that will be used to load the version model data
            kwargs: any number of key/values parameters, forwarded from `retrieve_version`
        Returns:
            model: the loaded model
        """
        raise NotImplementedError

    def __load(self, model_id, version_number, model_user_data, version_user_data, model_data_f, **kwargs):
        if (
            self.MODEL_USER_DATA_ADAPTER_CLASS_NAME_KEY in model_user_data
            and model_user_data[self.MODEL_USER_DATA_ADAPTER_CLASS_NAME_KEY] != self._adapter_class_name
        ):
            raise RuntimeError(
                f"Unable to load model '{model_id}@v{version_number}' with adapter '{self._adapter_class_name}': it was initially created by adapter '{model_user_data[self.MODEL_USER_DATA_ADAPTER_CLASS_NAME_KEY]}'"
            )

        return self._load(model_id, version_number, model_user_data, version_user_data, model_data_f, **kwargs)

    def _save(self, model, model_user_data, model_data_f, **kwargs):
        """
        Serialize and save a model
        Args:
            model: a model, as returned by the _create method of this class
            model_user_data (dict[str, str]): model user data
            model_data_f: file object that will be used to save the version model data
            kwargs: any number of key/values parameters, forwarded from `create_and_publish_initial_version` or `publish_version`
        Returns:
            version_user_data (dict[str, str]): additional version user data
        """
        raise NotImplementedError

    def __save(self, model, model_user_data, model_data_f, **kwargs):
        if (
            self.MODEL_USER_DATA_ADAPTER_CLASS_NAME_KEY in model_user_data
            and model_user_data[self.MODEL_USER_DATA_ADAPTER_CLASS_NAME_KEY] != self._adapter_class_name
        ):
            raise RuntimeError(
                f"Unable to save a new version of the model with adapter '{self._adapter_class_name}': it was initially created by adapter '{model_user_data[self.MODEL_USER_DATA_ADAPTER_CLASS_NAME_KEY]}'"
            )

        version_user_data = self._save(model, model_user_data, model_data_f, **kwargs)
        version_user_data[self.VERSION_USER_DATA_MODEL_CLASS_NAME_KEY] = get_full_class_name(model)

        return version_user_data

    async def create_and_publish_initial_version(self, model_id, **kwargs):
        """
        Create and publish to the model registry a model instance
        Parameters:
            model_id (string): unique identifier for the model
            kwargs: any number of key/values parameters, will be forwarded to `_create`
        Returns:
            model, model_info, version_info: the created model, its information and the information for the initial published version
        """

        # Create the agent model locally
        model, model_user_data = self.__create(model_id, **kwargs)

        # Create it in the model registry
        await self.get_model_registry_client().create_model(model_id, model_user_data)

        # Publish the first version
        version_info = await self.publish_version(model_id, model, **kwargs)
        return model, version_info

    async def publish_version(self, model_id, model, archived=False, **kwargs):
        """
        Publish to the model registry a new version of a model
        Parameters:
            model_id (string): unique identifier for the model
            model: a model, as returned by method of this class
            archive (bool - default is False): If true, the model version will be archived (i.e. stored in permanent storage)
        Returns:
            version_info: information for the initial published version
        """
        return await self.get_model_registry_client().publish_version(
            model_id=model_id, model=model, save_model=self.__save, archived=archived, **kwargs
        )

    async def retrieve_version(self, model_id, version_number=-1, **kwargs):
        """
        Publish to the model registry a new version of a model
        Parameters:
            model_id (string): Unique id of the model
            version_number (int - default is -1): The version number (-1 for the latest)
        Returns:
            model, model_info, version_info: A tuple containing the model, the model info and the model version info
        """
        return await self.get_model_registry_client().retrieve_version(
            model_id=model_id, load_model=self.__load, version_number=version_number, **kwargs
        )

    @abc.abstractmethod
    def _create_actor_implementations(self):
        """
        Create all the available actor implementation for this adapter
        Returns:
            dict[impl_name: string, (actor_impl: Callable, actor_classes: []string)]: key/value definition for the available actor implementations.
        """
        return {}

    @abc.abstractmethod
    def _create_run_implementations(self):
        """
        Create all the available run implementation for this adapter
        Returns:
            dict[impl_name: string, (sample_producer_impl: Callable, run_impl: Callable, default_run_config)]: key/value definition for the available run implementations.
        """
        return {}

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

        self.get_model_registry_client = context.get_model_registry_client
