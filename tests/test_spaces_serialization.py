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
import numpy as np
import pytest
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS, Overcooked, OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from pettingzoo.classic import connect_four_v3

from cogment_verse.specs.ndarray_serialization import SerializationFormat
from cogment_verse.specs.spaces_serialization import deserialize_gym_space, serialize_gym_space

# pylint: disable=no-member


def test_serialize_connect4_observation_space():
    env = connect_four_v3.env()
    env.reset()

    gym_space = env.observation_space("player_0")
    pb_space = serialize_gym_space(gym_space, serialization_format=SerializationFormat.STRUCTURED)

    assert len(pb_space.dict.spaces) == 2
    assert pb_space.dict.spaces[0].key == "action_mask"
    assert pb_space.dict.spaces[0].space.box.high.shape == [7]
    assert pb_space.dict.spaces[0].space.box.low.shape == [7]
    assert pb_space.dict.spaces[1].key == "observation"
    assert pb_space.dict.spaces[1].space.box.high.shape == [6, 7, 2]
    assert pb_space.dict.spaces[1].space.box.low.shape == [6, 7, 2]

    deserialized_space = deserialize_gym_space(pb_space)

    assert gym_space.shape == deserialized_space.shape


def test_serialize_cartpole_observation_space():
    env = gym.make("CartPole-v1", new_step_api=True)

    gym_space = env.observation_space

    pb_space = serialize_gym_space(gym_space, serialization_format=SerializationFormat.STRUCTURED)

    assert pb_space.box.high.shape == [4]
    assert pb_space.box.low.shape == [4]
    assert pb_space.box.high.double_data[0] == pytest.approx(4.8)
    assert pb_space.box.high.double_data[1] == np.finfo(np.float32).max

    deserialized_space = deserialize_gym_space(pb_space)

    assert gym_space.shape == deserialized_space.shape
    assert gym_space.low == pytest.approx(deserialized_space.low)
    assert gym_space.high == pytest.approx(deserialized_space.high)


def test_serialize_overcooked_observation_space():
    base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(base_mdp, **DEFAULT_ENV_PARAMS)
    gym_env = Overcooked(base_env=env, featurize_fn=env.featurize_state_mdp)
    gym_space = gym_env.observation_space

    pb_space = serialize_gym_space(gym_space, serialization_format=SerializationFormat.STRUCTURED)

    assert pb_space.box.high.shape == [96]
    assert pb_space.box.low.shape == [96]
    assert pb_space.box.low.double_data[0] == pytest.approx(0)
    assert pb_space.box.high.double_data[0] == np.inf

    deserialized_space = deserialize_gym_space(pb_space)

    assert gym_space.shape == deserialized_space.shape
    assert gym_space.low == pytest.approx(deserialized_space.low)
    assert gym_space.high == pytest.approx(deserialized_space.high)


def test_serialize_custom_observation_space():
    """Test serialization of gym spaces of type:
    Dict, Discrete, Box, MultiDiscrete, MultiBinary.
    """
    gym_space = Dict(
        {
            "ext_controller": MultiDiscrete([5, 2, 2]),
            "inner_state": Dict(
                {
                    "charge": Discrete(100),
                    "system_checks": MultiBinary(10),
                    "system_checks_seq": MultiBinary([2, 5, 10]),
                    "system_checks_array": MultiBinary(np.array([2, 5, 10])),
                    "job_status": Dict(
                        {
                            "task": Discrete(5),
                            "progress": Box(low=0, high=100, shape=()),
                        }
                    ),
                }
            ),
        }
    )

    pb_space = serialize_gym_space(gym_space, serialization_format=SerializationFormat.STRUCTURED)

    assert len(pb_space.dict.spaces) == 2
    assert pb_space.dict.spaces[0].key == "ext_controller"
    assert pb_space.dict.spaces[0].space.multi_discrete.nvec.shape == [3]

    assert pb_space.dict.spaces[1].key == "inner_state"
    assert len(pb_space.dict.spaces[1].space.dict.spaces) == 5

    assert pb_space.dict.spaces[1].space.dict.spaces[0].key == "charge"
    assert pb_space.dict.spaces[1].space.dict.spaces[0].space.discrete.n == 100

    assert pb_space.dict.spaces[1].space.dict.spaces[1].key == "job_status"
    assert len(pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces) == 2

    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[0].key == "progress"
    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[0].space.box.low.double_data[
        0
    ] == pytest.approx(0.0)
    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[0].space.box.high.double_data[
        0
    ] == pytest.approx(100.0)

    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[1].key == "task"
    assert pb_space.dict.spaces[1].space.dict.spaces[1].space.dict.spaces[1].space.discrete.n == 5

    assert pb_space.dict.spaces[1].space.dict.spaces[2].key == "system_checks"
    assert pb_space.dict.spaces[1].space.dict.spaces[2].space.multi_binary.n.shape == [1]

    assert pb_space.dict.spaces[1].space.dict.spaces[3].key == "system_checks_array"
    assert pb_space.dict.spaces[1].space.dict.spaces[3].space.multi_binary.n.shape == [3]

    assert pb_space.dict.spaces[1].space.dict.spaces[4].key == "system_checks_seq"
    assert pb_space.dict.spaces[1].space.dict.spaces[4].space.multi_binary.n.shape == [3]
