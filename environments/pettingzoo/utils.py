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

from enum import Enum
import re
from typing import List


class PettingZooEnvType(Enum):
    ATARI = "atari"
    CLASSIC = "classic"
    MPE = "mpe"


class MpeSpecType(Enum):
    """ Used to associate different environment specs to different actors.
    """
    DEFAULT = "player"
    AGENT = "mpe_agent"
    ADVERSARY = "mpe_adversary"

    @classmethod
    def from_config(cls, spec_type_str: str):
        try:
            return cls(spec_type_str)
        except ValueError:
            raise ValueError(f"Actor specs type ({spec_type_str}) is not a supported type: [{', '.join(MpeSpecType.values)}]")

    @classmethod
    @property
    def values(self) -> List[str]:
        """ Return list of all values available in the enum """
        return list(spec_type.value for spec_type in MpeSpecType)


def is_homogeneous(env) -> bool:
    num_players = 0
    observation_space = None
    action_space = None
    for player in env.possible_agents:
        num_players += 1
        if observation_space is None:
            observation_space = env.observation_space(player)
            action_space = env.action_space(player)
        else:
            if observation_space != env.observation_space(player) or action_space != env.action_space(player):
                return False
    return True

def get_strings_with_prefix(strings, prefix):
    pattern = r'^' + prefix
    matches = [string for string in strings if re.match(pattern, string)]
    return matches


