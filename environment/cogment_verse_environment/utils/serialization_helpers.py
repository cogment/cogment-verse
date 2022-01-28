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

import cv2
import numpy as np
from data_pb2 import NDArray


def deserialize_np_array(nd_array):
    return np.frombuffer(nd_array.data, dtype=nd_array.dtype).reshape(*nd_array.shape)


def serialize_np_array(np_array):
    return NDArray(shape=np_array.shape, dtype=str(np_array.dtype), data=np_array.tobytes())


def deserialize_img(img_bytes):
    return cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def serialize_img(img):
    # note rgb -> bgr for cv2
    result, data = cv2.imencode(".jpg", img[:, :, ::-1])
    assert result
    return data.tobytes()
