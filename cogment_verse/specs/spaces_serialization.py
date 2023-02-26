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

import gym
import numpy as np
from ndarray_pb2 import DTYPE_INT8
from spaces_pb2 import (Box, Dict, Discrete,  # pylint: disable=import-error
                        MultiBinary, MultiDiscrete, Space)

from .ndarray_serialization import (SerializationFormat, deserialize_ndarray,
                                    serialize_ndarray)


def serialize_gym_space(gym_space, serilization_format=SerializationFormat.STRUCTURED):
    if isinstance(gym_space, gym.spaces.Discrete):
        return Space(discrete=Discrete(n=gym_space.n, start=gym_space.start))

    if isinstance(gym_space, gym.spaces.Box):
        low = gym_space.low
        high = gym_space.high
        return Space(
            box=Box(
                low=serialize_ndarray(low, serilization_format=serilization_format),
                high=serialize_ndarray(high, serilization_format=serilization_format),
            )
        )

    if isinstance(gym_space, gym.spaces.MultiBinary):
        if isinstance(gym_space.n, np.ndarray):
            n = gym_space.n
        elif isinstance(gym_space.n, int):
            n = np.array([gym_space.n], dtype=np.dtype("int8"))
        else:
            n = np.array(gym_space.n, dtype=np.dtype("int8"))
        return Space(
            multi_binary=MultiBinary(n=serialize_ndarray(n, serilization_format=serilization_format))
        )

    if isinstance(gym_space, gym.spaces.MultiDiscrete):
        nvec = gym_space.nvec
        return Space(
            multi_discrete=MultiDiscrete(
                nvec=serialize_ndarray(nvec, serilization_format=serilization_format)
            )
        )

    if isinstance(gym_space, gym.spaces.Dict):
        spaces = []
        for key, gym_sub_space in gym_space.spaces.items():
            spaces.append(Dict.SubSpace(key=key, space=serialize_gym_space(gym_sub_space)))
        return Space(dict=Dict(spaces=spaces))
    raise RuntimeError(f"[{type(gym_space)}] is not a supported space type")


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
        n = deserialize_ndarray(multi_binary_space_pb.n)
        if n.size > 1:
            return gym.spaces.MultiBinary(n=n)
        else:
            return gym.spaces.MultiBinary(n=n[0])
    if space_kind == "multi_discrete":
        multi_discrete_space_pb = pb_space.multi_discrete
        nvec = deserialize_ndarray(multi_discrete_space_pb.nvec)
        return gym.spaces.MultiBinary(nvec=nvec)
    if space_kind == "dict":
        dict_space_pb = pb_space.dict
        spaces = []
        for sub_space in dict_space_pb.spaces:
            spaces.append((sub_space.key, deserialize_gym_space(sub_space.space)))

        return gym.spaces.Dict(spaces=spaces)

    raise RuntimeError(f"[{space_kind}] is not a supported space type")
