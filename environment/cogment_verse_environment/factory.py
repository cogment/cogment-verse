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

from cogment_verse_environment.gym_env import GymEnv
from cogment_verse_environment.atari import AtariEnv
from cogment_verse_environment.tetris import TetrisEnv
from cogment_verse_environment.minatarenv import MinAtarEnv
from cogment_verse_environment.zoo_env import PettingZooEnv
from pipe_world.pipe_world import PipeWorld

ENVIRONMENT_CONSTRUCTORS = {
    "gym": GymEnv,
    "atari": AtariEnv,
    "minatar": MinAtarEnv,
    "tetris": TetrisEnv,
    "pettingzoo": PettingZooEnv,
    "pipe_world": PipeWorld
}


def make_environment(env_type, env_name, **kwargs):
    print("In make_environment")
    return ENVIRONMENT_CONSTRUCTORS[env_type](env_name=env_name, **kwargs)
