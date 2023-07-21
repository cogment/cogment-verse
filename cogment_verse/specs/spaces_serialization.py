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

import gym
import gymnasium
import numpy as np
from spaces_pb2 import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Type  # pylint: disable=import-error

from .ndarray_serialization import SerializationFormat, deserialize_ndarray, serialize_ndarray


def serialize_gym_space(space, serialization_format=SerializationFormat.STRUCTURED):

    if isinstance(space, gym.spaces.space.Space):
        space_type = Type.GYM
    elif isinstance(space, gymnasium.spaces.Space):
        space_type = Type.GYMNASIUM
    else:
        raise ValueError("Unsupported space type")

    if isinstance(space, (gym.spaces.Discrete, gymnasium.spaces.Discrete)):
        return Space(type=space_type, discrete=Discrete(n=space.n, start=space.start))
    if isinstance(space, (gym.spaces.Box, gymnasium.spaces.Box)):
        low = space.low
        high = space.high
        return Space(
            type=space_type,
            box=Box(
                low=serialize_ndarray(low, serialization_format=serialization_format),
                high=serialize_ndarray(high, serialization_format=serialization_format),
            )
        )

    if isinstance(space, (gym.spaces.MultiBinary, gymnasium.spaces.MultiBinary)):
        if isinstance(space.n, np.ndarray):
            size = space.n
        elif isinstance(space.n, int):
            size = np.array([space.n], dtype=np.dtype("int32"))
        else:
            size = np.array(space.n, dtype=np.dtype("int32"))
        return Space(
            type=space_type,
            multi_binary=MultiBinary(n=serialize_ndarray(size, serialization_format=serialization_format))
        )

    if isinstance(space, (gym.spaces.MultiDiscrete, gymnasium.spaces.MultiDiscrete)):
        nvec = space.nvec
        return Space(
            type=space_type,
            multi_discrete=MultiDiscrete(nvec=serialize_ndarray(nvec, serialization_format=serialization_format))
        )

    if isinstance(space, (gym.spaces.Dict, gymnasium.spaces.Dict)):
        spaces = []
        for key, gym_sub_space in space.spaces.items():
            spaces.append(Dict.SubSpace(key=key, space=serialize_gym_space(gym_sub_space)))
        return Space(type=space_type, dict=Dict(spaces=spaces))
    raise RuntimeError(f"[{type(space)}] is not a supported space type")


def deserialize_space(pb_space):
    space_type = pb_space.type
    if space_type == Type.GYM:
        return deserialize_gym_space(pb_space)
    elif space_type == Type.GYMNASIUM:
        return deserialize_gymnasium_space(pb_space)
    else:
        raise RuntimeError(f"[{space_type}] is not a supported space type")


def deserialize_gym_space(pb_space):
    space_kind = pb_space.WhichOneof("kind")
    if space_kind == "discrete":
        discrete_space_pb = pb_space.discrete
        return gym.spaces.Discrete(n=discrete_space_pb.n, start=discrete_space_pb.start)
    if space_kind == "box":
        box_space_pb = pb_space.box
        low = deserialize_ndarray(box_space_pb.low)
        high = deserialize_ndarray(box_space_pb.high)
        return gym.spaces.Box(low=low, high=high, shape=low.shape, dtype=low.dtype)
    if space_kind == "multi_binary":
        multi_binary_space_pb = pb_space.multi_binary
        size = deserialize_ndarray(multi_binary_space_pb.n)
        if size.size > 1:
            return gym.spaces.MultiBinary(n=size)
        return gym.spaces.MultiBinary(n=size[0])
    if space_kind == "multi_discrete":
        multi_discrete_space_pb = pb_space.multi_discrete
        nvec = deserialize_ndarray(multi_discrete_space_pb.nvec)
        return gym.spaces.MultiDiscrete(nvec=nvec)
    if space_kind == "dict":
        dict_space_pb = pb_space.dict
        spaces = []
        for sub_space in dict_space_pb.spaces:
            spaces.append((sub_space.key, deserialize_gym_space(sub_space.space)))

        return gym.spaces.Dict(spaces=spaces)

    raise RuntimeError(f"[{space_kind}] is not a supported space kind")


def deserialize_gymnasium_space(pb_space):
    space_kind = pb_space.WhichOneof("kind")
    if space_kind == "discrete":
        discrete_space_pb = pb_space.discrete
        return gymnasium.spaces.Discrete(n=discrete_space_pb.n, start=discrete_space_pb.start)
    if space_kind == "box":
        box_space_pb = pb_space.box
        low = deserialize_ndarray(box_space_pb.low)
        high = deserialize_ndarray(box_space_pb.high)
        return gymnasium.spaces.Box(low=low, high=high, shape=low.shape, dtype=low.dtype)
    if space_kind == "multi_binary":
        multi_binary_space_pb = pb_space.multi_binary
        size = deserialize_ndarray(multi_binary_space_pb.n)
        if size.size > 1:
            return gymnasium.spaces.MultiBinary(n=size)
        return gymnasium.spaces.MultiBinary(n=size[0])
    if space_kind == "multi_discrete":
        multi_discrete_space_pb = pb_space.multi_discrete
        nvec = deserialize_ndarray(multi_discrete_space_pb.nvec)
        return gymnasium.spaces.MultiDiscrete(nvec=nvec)
    if space_kind == "dict":
        dict_space_pb = pb_space.dict
        spaces = []
        for sub_space in dict_space_pb.spaces:
            spaces.append((sub_space.key, deserialize_gymnasium_space(sub_space.space)))

        return gymnasium.spaces.Dict(spaces=spaces)

    raise RuntimeError(f"[{space_kind}] is not a supported space kind")


def flatten(space, observation):
    if isinstance(space, gym.spaces.space.Space):
        return gym.spaces.flatten(space, observation)
    elif isinstance(space, gymnasium.spaces.Space):
        return gymnasium.spaces.flatten(space, observation)
    else:
        raise ValueError("Unsupported space type")


def unflatten(original_space, flat_observation):
    if isinstance(original_space, gym.spaces.space.Space):
        return gym.spaces.unflatten(original_space, flat_observation)
    elif isinstance(original_space, gymnasium.spaces.Space):
        return gymnasium.spaces.unflatten(original_space, flat_observation)
    else:
        raise ValueError("Unsupported space type")

