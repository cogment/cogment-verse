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

# pylint: disable=broad-except

from __future__ import annotations

import abc


class Model:
    def __init__(self, model_id, iteration=0):
        self.model_id = model_id
        self.iteration = iteration

    @abc.abstractmethod
    def get_model_user_data(self):
        """
        Retrieve the user data associated with the model instance
        Returns:
            model_user_data (dict[str, str]): model user data that will be saved alongside the model
        """
        return {}

    @staticmethod
    @abc.abstractmethod
    def serialize_model(model):
        """
        Load a serialized model instance and return it
        Args:
            model: file object that will be used to save the version model data
        Returns:
            model: the serialized model
        """

    @classmethod
    @abc.abstractmethod
    def deserialize_model(cls, serialized_model):
        """
        Serialize and save the model
        Args:
            serialized_model: file object that will be used to load the version model data
        Returns:
            model: the deserialized model
        """

    @classmethod
    async def retrieve_model(cls, model_registry, model_id, iteration) -> Model:
        """
        Retrieve and deserialize a specific or latest model iteration from the model registry.
        If the configuration is set to retrieve iteration -1, the latest model is tracked.
        Otherwise, it retrieves the specific model iteration.
        Args:
            model_registry: model registry
            model_id: model id to retrieve
            iteration: model iteration to retrieve
        Returns:
            model: the deserialized model
        """
        if iteration == -1:
            latest_model = await model_registry.track_latest_model(
                name=model_id, deserialize_func=cls.deserialize_model
            )
            model, iteration_info = await latest_model.get()
        else:
            serialized_model = await model_registry.retrieve_model(model_id, iteration)
            model = cls.deserialize_model(serialized_model)
            iteration_info = await model_registry.get_iteration_info(model_id, iteration)
        model.iteration = iteration_info.iteration
        return model
