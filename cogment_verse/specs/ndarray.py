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

import numpy as np
from data_pb2 import NDArray  # pylint: disable=E0611


def deserialize_ndarray(nd_array):
    return np.frombuffer(nd_array.data, dtype=nd_array.dtype).reshape(*nd_array.shape)


def serialize_ndarray(nd_array):
    return NDArray(shape=nd_array.shape, dtype=str(nd_array.dtype), data=nd_array.tobytes())


def create_one_hot_ndarray(values, size):
    nd_array = np.zeros(size)
    nd_array[values] = 1
    return nd_array
