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

import gym
import numpy as np
import yaml
from minatar.environment import Environment as MinatarEnvironment

env_specs = {}


def gym_spaces_discrete_representer(dumper, discrete_space):
    return dumper.represent_mapping("gym.spaces.Discrete", {"n": discrete_space.n})


def gym_spaces_box_representer(dumper, box_space):
    first_low = np.ravel(box_space.low)[0]
    first_high = np.ravel(box_space.high)[0]
    return dumper.represent_mapping(
        "gym.spaces.Box",
        {
            "low": first_low.item() if np.all(box_space.low == first_low) else box_space.low.tolist(),
            "high": first_high.item() if np.all(box_space.high == first_high) else box_space.high.tolist(),
            "shape": box_space.shape,
            "dtype": box_space.dtype.name,
        },
    )


yaml.add_representer(gym.spaces.Discrete, gym_spaces_discrete_representer)
yaml.add_representer(gym.spaces.Box, gym_spaces_box_representer)

## Gym environments
for gym_env_spec in gym.envs.registry.all():
    if (
        gym_env_spec.entry_point != "gym.envs.atari:AtariEnv"
        and not gym_env_spec.entry_point.startswith("gym.envs.classic_control")
        and not gym_env_spec.entry_point.startswith("gym.envs.box2d")
    ):
        # Skipping we don't care about other
        continue
    if (
        gym_env_spec.entry_point == "gym.envs.atari:AtariEnv"
        and (gym_env_spec.id.endswith("v4") or gym_env_spec.id.endswith("v0"))
        and (gym_env_spec.id.find("-ram") != -1 or gym_env_spec.id.find("NoFrameskip") == -1)
    ):
        # Skipping we only want no frameskip environments with either sticky_actions or not
        continue

    try:
        gym_env = gym_env_spec.make()
    except gym.error.DependencyNotInstalled:
        # Skipping, a required depency is missing
        continue

    env_specs[f"gym/{gym_env_spec.id}"] = {
        "class_name": gym_env_spec.entry_point,
        "reward_threshold": gym_env_spec.reward_threshold,
        "max_episode_steps": gym_env_spec.max_episode_steps,
        "sticky_actions": gym_env_spec.entry_point == "gym.envs.atari:AtariEnv" and gym_env_spec.id.endswith("v4"),
        "action_space": gym_env.action_space,
        "observation_space": gym_env.observation_space,
        "agents": ["player"],
    }

for minatar_env_id in (
    "asterix",
    "breakout",
    "freeway",
    "seaquest",
    "space_invaders",
):
    minatar_env = MinatarEnvironment(env_name=minatar_env_id)
    minatar_state_shape = minatar_env.state_shape()
    env_specs[f"minatar/{minatar_env_id}"] = {
        "class_name": "minatar.environment.Environment",
        "reward_threshold": None,
        "max_episode_steps": None,
        "action_space": gym.spaces.Discrete(len(minatar_env.minimal_action_set())),
        "observation_space": gym.spaces.Box(
            low=0,
            high=minatar_state_shape[2],
            shape=(minatar_state_shape[0], minatar_state_shape[1]),
            dtype=np.uint8,
        ),
        "agents": ["player"],
    }

print(yaml.dump(env_specs))
print(f"{len(env_specs)} environments available")
