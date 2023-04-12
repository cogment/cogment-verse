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

import abc


class Model:
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
    def deserialize_model(cls, serialized_model, model_id, version_number):
        """
        Serialize and save the model
        Args:
            serialized_model: file object that will be used to load the version model data
            model_id (string): unique identifier for the model
            version_number (int): unique identifier for the model version
        Returns:
            model: the deserialized model
        """