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

from data_pb2 import Space, SpaceMask, SpaceValue  # pylint: disable=import-error

from .value import create_space_values
from .ndarray import deserialize_ndarray, serialize_ndarray, create_one_hot_ndarray


def space_prop_flattened_dimensions(prop):
    """
    Computes the number of dimensions a flattened equivalent of the given space property would have.
    """
    assert isinstance(prop, Space.Property)

    if prop.WhichOneof("type") == "discrete":
        return max(len(prop.discrete.labels), prop.discrete.num)
    # box
    if len(prop.box.shape) == 0:
        return 0
    prop_dim = 1
    for dim in prop.box.shape:
        prop_dim *= dim
    return prop_dim


def flattened_dimensions(space):
    """
    Computes the number of dimensions a flattened equivalent of the given space would have.
    """
    assert isinstance(space, Space)

    space_dim = 0
    for prop in space.properties:
        space_dim += space_prop_flattened_dimensions(prop)

    return space_dim


def flatten_prop(space, value, prop_idx):
    assert isinstance(space, Space)
    assert isinstance(value, SpaceValue)

    if prop_idx >= len(value.properties):
        # If a value defines less properties than its space we make the remaining zeros
        return np.zeros(space_prop_flattened_dimensions(space.properties[prop_idx]))
    prop_value = value.properties[prop_idx]
    value_type = prop_value.WhichOneof("value")
    if value_type == "discrete":
        space_prop = space.properties[prop_idx]
        return create_one_hot_ndarray(
            [prop_value.discrete], max(len(space_prop.discrete.labels), space_prop.discrete.num)
        )
    if value_type == "box":
        return deserialize_ndarray(prop_value.box).flatten()
    # if value_type == "simple_box":
    return np.array(prop_value.simple_box.values)


def flatten(space, value):
    assert isinstance(space, Space)
    assert isinstance(value, SpaceValue)

    return np.concatenate([flatten_prop(space, value, prop_idx) for prop_idx in range(len(space.properties))])


def unflatten(space, flat_value):
    assert isinstance(space, Space)
    assert len(np.shape(flat_value)) == 1

    [value] = create_space_values(space)
    flat_value_idx = 0
    for prop_idx, prop in enumerate(space.properties):
        flat_value_idx_end = flat_value_idx + space_prop_flattened_dimensions(prop)
        flat_value_prop = flat_value[flat_value_idx:flat_value_idx_end]
        if prop.WhichOneof("type") == "discrete":
            # np.nonzero on a flat array returns a 1D tuple of the indices whose value is != 0
            value.properties[prop_idx].discrete = np.nonzero(flat_value_prop)[0][0]
        else:
            value.properties[prop_idx].box.CopyFrom(serialize_ndarray(flat_value_prop.reshape(prop.box.shape)))

        flat_value_idx = flat_value_idx_end

    return value


def flatten_mask_prop(space, mask, prop_idx):
    assert isinstance(space, Space)
    assert isinstance(mask, SpaceMask)

    space_prop = space.properties[prop_idx]
    if prop_idx >= len(mask.properties):
        # If a mask defines less properties than its space we make the remaining ones
        return np.ones(space_prop_flattened_dimensions(space_prop))

    value_type = space_prop.WhichOneof("type")
    prop_mask = mask.properties[prop_idx]
    if value_type == "discrete":
        return create_one_hot_ndarray(prop_mask.discrete, max(len(space_prop.discrete.labels), space_prop.discrete.num))
    # if value_type == "box":
    return np.ones(space_prop_flattened_dimensions(space_prop))


def flatten_mask(space, mask):
    assert isinstance(space, Space)
    assert isinstance(mask, SpaceMask)

    return np.concatenate([flatten_mask_prop(space, mask, prop_idx) for prop_idx in range(len(space.properties))])
