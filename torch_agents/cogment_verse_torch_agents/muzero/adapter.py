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

from data_pb2 import (
    MuZeroTrainingRunConfig,
    MuZeroTrainingConfig,
    AgentAction,
    TrialConfig,
    TrialActor,
    EnvConfig,
    ActorConfig,
    MLPNetworkConfig,
)

from cogment_verse import AgentAdapter
from cogment_verse import MlflowExperimentTracker

from cogment.api.common_pb2 import TrialState
import cogment

import logging
import torch
import numpy as np

from collections import namedtuple

log = logging.getLogger(__name__)

from .muzero import MuZero

# pylint: disable=arguments-differ


class MuZeroAgentAdapter(AgentAdapter):
    def __init__(self):
        super().__init__()

    def _create(self, model_id, **kwargs):
        """
        Create and return a model instance
        Parameters:
            model_id (string): unique identifier for the model
            kwargs: any number of key/values paramters
        Returns:
            model: the created model
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def _save(self, model, model_data_f):
        """
        Serialize and save a model
        Args:
            model: a model, as returned by method of this class
            model_data_f: file object that will be used to save the version model data
        Returns:
            version_user_data (dict[str, str]): version user data
        """
        raise NotImplementedError

    def _create_actor_implementations(self):
        """
        Create all the available actor implementation for this adapter
        Returns:
            dict[impl_name: string, (actor_impl: Callable, actor_classes: []string)]: key/value definition for the available actor implementations.
        """
        raise NotImplementedError

    def _create_run_implementations(self):
        """
        Create all the available run implementation for this adapter
        Returns:
            dict[impl_name: string, (sample_producer_impl: Callable, run_impl: Callable, default_run_config)]: key/value definition for the available run implementations.
        """
        raise NotImplementedError
