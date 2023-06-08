# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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
import pytest
from google.protobuf.json_format import MessageToDict, ParseDict

from cogment_verse.specs.ndarray_serialization import Array, SerializationFormat, deserialize_ndarray, serialize_ndarray


def create_large_nd_array(seed=12, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.random((10, 10, 10), dtype=dtype) * 1000 - 500


def create_large_int_nd_array(seed=12, dtype=np.int32):
    np.random.seed(seed)
    return np.random.randint(-500, 500, (10, 10, 10), dtype=dtype)


def create_large_uint_nd_array(seed=12, dtype=np.uint32):
    np.random.seed(seed)
    return np.random.randint(0, 250, (10, 10, 10), dtype=dtype)


@pytest.mark.benchmark(group="serialize_ndarray")
def test_serialize_ndarray_raw(benchmark):
    array = create_large_int_nd_array()
    serialized_array = benchmark(serialize_ndarray, array, SerializationFormat.RAW)

    deserialized_array = deserialize_ndarray(serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="serialize_ndarray")
def test_serialize_ndarray_raw_int32(benchmark):
    array = create_large_int_nd_array(seed=12, dtype=np.int32)
    serialized_array = benchmark(serialize_ndarray, array, SerializationFormat.RAW)

    deserialized_array = deserialize_ndarray(serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="serialize_ndarray")
def test_serialize_ndarray_npy(benchmark):
    array = create_large_nd_array()
    serialized_array = benchmark(serialize_ndarray, array, SerializationFormat.NPY)

    deserialized_array = deserialize_ndarray(serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="serialize_ndarray")
def test_serialize_ndarray_npy_int32(benchmark):
    array = create_large_int_nd_array(seed=12, dtype=np.int32)
    serialized_array = benchmark(serialize_ndarray, array, SerializationFormat.NPY)

    deserialized_array = deserialize_ndarray(serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="serialize_ndarray")
def test_serialize_ndarray_structured(benchmark):
    array = create_large_nd_array()
    serialized_array = benchmark(serialize_ndarray, array, SerializationFormat.STRUCTURED)

    deserialized_array = deserialize_ndarray(serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="serialize_ndarray")
def test_serialize_ndarray_structured_int32(benchmark):
    array = create_large_int_nd_array(seed=12, dtype=np.int32)
    serialized_array = benchmark(serialize_ndarray, array, SerializationFormat.STRUCTURED)

    deserialized_array = deserialize_ndarray(serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="serialize_ndarray")
def test_serialize_ndarray_structured_uint8(benchmark):
    array = create_large_uint_nd_array(seed=67, dtype=np.uint8)
    serialized_array = benchmark(serialize_ndarray, array, SerializationFormat.STRUCTURED)

    deserialized_array = deserialize_ndarray(serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="deserialize_ndarray")
def test_deserialize_ndarray_raw(benchmark):
    array = create_large_nd_array()
    serialized_array = serialize_ndarray(array, SerializationFormat.RAW)
    deserialized_array = benchmark(deserialize_ndarray, serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="deserialize_ndarray")
def test_deserialize_ndarray_npy(benchmark):
    array = create_large_nd_array()
    serialized_array = serialize_ndarray(array, SerializationFormat.NPY)
    deserialized_array = benchmark(deserialize_ndarray, serialized_array)
    assert np.array_equal(array, deserialized_array)


@pytest.mark.benchmark(group="deserialize_ndarray")
def test_deserialize_ndarray_structured(benchmark):
    array = create_large_nd_array()
    serialized_array = serialize_ndarray(array, SerializationFormat.STRUCTURED)
    deserialized_array = benchmark(deserialize_ndarray, serialized_array)
    assert np.array_equal(array, deserialized_array)


def test_serialize_ndarray_with_npinf_through_dict():
    array = np.array([[-np.inf, 2], [np.inf, 4]])
    pb_array = serialize_ndarray(array, SerializationFormat.STRUCTURED)
    pb_array_dict = MessageToDict(pb_array, preserving_proto_field_name=True)
    deserialized_pb_array = ParseDict(pb_array_dict, Array())
    deserialized_array = deserialize_ndarray(deserialized_pb_array)
    assert np.array_equal(array, deserialized_array)


def test_serialize_nd_array_one_value():
    array = np.array(12)
    pb_array = serialize_ndarray(array)
    deserialized_array = deserialize_ndarray(pb_array)
    assert np.array_equal(array, deserialized_array)
