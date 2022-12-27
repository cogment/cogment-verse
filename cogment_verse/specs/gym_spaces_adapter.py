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
import gymnasium as gymna

from data_pb2 import Space, SpaceValue  # pylint: disable=import-error

from .ndarray import deserialize_ndarray, serialize_ndarray

SPACES_BOUND_MAX = float.fromhex("0x1.fffffep+127")
SPACES_BOUND_MIN = -SPACES_BOUND_MAX


def gym_space_from_space(space):
    gym_spaces_dict = {}
    for prop in space.properties:
        prop_type = prop.WhichOneof("type")
        if prop_type == "discrete":
            num_action = max(len(prop.discrete.labels), prop.discrete.num)
            gym_spaces_dict[prop.key] = gym.spaces.Discrete(num_action)
        if prop_type == "box":
            gym_spaces_dict[prop.key] = gym.spaces.Box(
                low=np.array([bound.bound if bound.bound is not None else -np.inf for bound in prop.box.low]),
                high=np.array([bound.bound if bound.bound is not None else np.inf for bound in prop.box.high]),
                shape=prop.box.shape,
            )
    if len(gym_spaces_dict) == 1:
        return next(iter(gym_spaces_dict.values()))
    return gym.spaces.Dict(gym_spaces_dict)


def space_properties_from_gym_space(gym_space):
    if isinstance(gym_space, (gym.spaces.Box, gymna.spaces.Box)):
        return [
            Space.Property(
                box=Space.Box(
                    shape=list(gym_space.shape),
                    low=[
                        Space.Bound(bound=v) if SPACES_BOUND_MIN < v < SPACES_BOUND_MAX else Space.Bound()
                        for v in gym_space.low.flat
                    ],
                    high=[
                        Space.Bound(bound=v) if SPACES_BOUND_MIN < v < SPACES_BOUND_MAX else Space.Bound()
                        for v in gym_space.high.flat
                    ],
                )
            )
        ]
    if isinstance(gym_space, (gym.spaces.Discrete, gymna.spaces.Discrete)):
        return [Space.Property(discrete=Space.Discrete(num=gym_space.n))]

    if isinstance(gym_space, (gym.spaces.Dict, gymna.spaces.Dict)):
        properties = []
        for prop_key, gym_sub_space in gym_space.properties:
            for sub_prop in space_properties_from_gym_space(gym_sub_space):
                sub_prop.key = (
                    f"{prop_key}.{sub_prop.key}" if sub_prop.key is not None else prop_key  # pylint: disable=no-member
                )
                properties.append(sub_prop)
        return properties
    raise RuntimeError(f"[{type(gym_space)}] is not a supported gym space type")


def space_from_gym_space(gym_space):
    return Space(properties=space_properties_from_gym_space(gym_space))


def gym_action_from_action(space, action):
    if len(action.properties) == 1:
        prop_value = action.properties[0]
        value_type = prop_value.WhichOneof("value")
        if value_type == "discrete":
            return prop_value.discrete
        if value_type == "box":
            return deserialize_ndarray(prop_value.box)
        # value_type == "simple_box"
        return np.array(prop_value.simple_box.values).reshape(space.properties[0].box.shape)
    raise RuntimeError(f"Not supporting spaces not having one property, got {len(action.properties)}")


def observation_from_gym_observation(gym_space, gym_observation):
    if isinstance(gym_space, (gym.spaces.Box, gymna.spaces.Box)):
        return SpaceValue(properties=[SpaceValue.PropertyValue(box=serialize_ndarray(gym_observation))])
    if isinstance(gym_space, (gym.spaces.Discrete, gymna.spaces.Discrete)):
        return SpaceValue(properties=[SpaceValue.PropertyValue(discrete=gym_observation.item())])
    raise RuntimeError(f"[{type(gym_space)}] is not a supported gym space type")
