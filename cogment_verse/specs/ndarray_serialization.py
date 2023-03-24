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

from enum import Enum
import io

import numpy as np

# pylint: disable=import-error
from ndarray_pb2 import (
    Array,
    DTYPE_UNKNOWN,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
)

PB_DTYPE_FROM_DTYPE = {
    "float32": DTYPE_FLOAT32,
    "float64": DTYPE_FLOAT64,
    "int8": DTYPE_INT8,
    "int32": DTYPE_INT32,
    "int64": DTYPE_INT64,
    "uint8": DTYPE_UINT8,
}

DTYPE_FROM_PB_DTYPE = {
    DTYPE_FLOAT32: np.dtype("float32"),
    DTYPE_FLOAT64: np.dtype("float64"),
    DTYPE_INT8: np.dtype("int8"),
    DTYPE_INT32: np.dtype("int32"),
    DTYPE_INT64: np.dtype("int64"),
    DTYPE_UINT8: np.dtype("uint8"),
}

DOUBLE_DTYPES = frozenset(["float32", "float64"])
INT32_DTYPES = frozenset(["int8", "int32"])
INT64_DTYPES = frozenset(["int64"])


class SerializationFormat(Enum):
    RAW = 1
    NPY = 2
    STRUCTURED = 3


def serialize_ndarray(nd_array, serilization_format=SerializationFormat.RAW):
    str_dtype = str(nd_array.dtype)
    pb_dtype = PB_DTYPE_FROM_DTYPE.get(str_dtype, DTYPE_UNKNOWN)

    # SerializationFormat.RAW
    if serilization_format is SerializationFormat.RAW:
        return Array(
            shape=nd_array.shape,
            dtype=pb_dtype,
            raw_data=nd_array.tobytes(order="C"),
        )

    # SerializationFormat.NPY
    if serilization_format is SerializationFormat.NPY:
        buffer = io.BytesIO()
        np.save(buffer, nd_array, allow_pickle=False)
        return Array(
            shape=nd_array.shape,
            dtype=pb_dtype,
            npy_data=buffer.getvalue(),
        )

    # SerializationFormat.STRUCTURED:
    if str_dtype in DOUBLE_DTYPES:
        return Array(
            shape=nd_array.shape,
            dtype=pb_dtype,
            double_data=nd_array.ravel(order="C").tolist(),
        )
    if str_dtype in INT32_DTYPES:
        return Array(
            shape=nd_array.shape,
            dtype=pb_dtype,
            int32_data=nd_array.ravel(order="C").tolist(),
        )
    if str_dtype in INT64_DTYPES:
        return Array(
            shape=nd_array.shape,
            dtype=pb_dtype,
            int64_data=nd_array.ravel(order="C").tolist(),
        )

    raise RuntimeError(f"[{str_dtype}] is not a supported numpy dtype for serialization format [{format}]")


def deserialize_ndarray(pb_array):
    dtype = DTYPE_FROM_PB_DTYPE.get(pb_array.dtype)
    str_dtype = str(dtype)
    shape = tuple(pb_array.shape)

    if len(pb_array.raw_data) > 0:
        return np.frombuffer(pb_array.raw_data, dtype=dtype).reshape(shape, order="C")

    if len(pb_array.npy_data) > 0:
        buffer = io.BytesIO(pb_array.npy_data)
        return np.load(buffer, allow_pickle=False)

    # SerializationFormat.STRUCTURED
    if str_dtype in DOUBLE_DTYPES:
        return np.array(pb_array.double_data, dtype=dtype).reshape(shape, order="C")
    if str_dtype in INT32_DTYPES:
        return np.array(pb_array.int32_data, dtype=dtype).reshape(shape, order="C")
    if str_dtype in INT64_DTYPES:
        return np.array(pb_array.int64_data, dtype=dtype).reshape(shape, order="C")

    raise RuntimeError(f"[{str_dtype}] is not a supported numpy dtype for serialization format [{format}]")
