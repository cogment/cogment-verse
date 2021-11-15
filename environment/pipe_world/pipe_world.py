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

from cogment_verse_environment.base import BaseEnv, GymObservation
from cogment_verse_environment.env_spec import EnvSpec

from pipe_world.logical_segment import LogicalSegments
import numpy as np


class PipeWorld(BaseEnv):
    def __init__(self, *, env_name, num_players=1, framestack=1, **kwargs):
        self.logical_segments = LogicalSegments()
        self.score = 100000.0
        self.budget = 1000.0
        self.tick_id = 0
        self.seed_number = 0
        super().__init__(
            env_spec=self.create_env_spec(env_name, **kwargs), num_players=num_players, framestack=1
        )

    def create_env_spec(self, env_name, **_kwargs):
        act_dim = 600
        return EnvSpec(
            env_name=env_name,
            obs_dim=(300, 4),
            act_dim=[act_dim],
            act_shape=(300, 2),
        )

    def reset(self):
        self.logical_segments = LogicalSegments()
        self.score = 100000.0
        self.budget = 1000.0
        self.tick_id = 0
        self.scale_segments(0.0)

        return GymObservation(
            observation=self.generate_observation(),
            rewards=[0.0],
            current_player=self._turn,
            legal_moves_as_int=[],
            done=False,
            info={},
        )

    def step(self, action):
        self.tick_id += 1
        if self.tick_id % 20 == 0:
            self.budget += 1000.0
        cost = self.logical_segments.step()
        self.score -= cost

        done = self.score < 0.0

        GymObservation(
            observation=self.generate_observation(),
            rewards=[-cost],
            current_player=self._turn,
            legal_moves_as_int=[],
            done=done,
            info={},
        )

    def render(self, mode="rgb_array"):
        return np.array([[[0, 0, 0]]], dtype=np.uint8)

    def seed(self, seed=None):
        self.seed_number = seed

    def maintain(self, logical_segment_index):
        logical_segment = self.logical_segments.logical_segments[logical_segment_index]
        if self.budget >= logical_segment.cost_of_maintnance:
            self.budget -= logical_segment.cost_of_maintnance
            logical_segment.maintain()

    def inspect(self, logical_segment_index):
        logical_segment = self.logical_segments.logical_segments[logical_segment_index]
        self.budget -= logical_segment.cost_of_inspection
        logical_segment.inspect()

    def scale_segments(self, scale):
        self.logical_segments.scale_segments(scale)

    def display(self):
        self.logical_segments.display()

    def generate_observation(self):
        return self.logical_segments.generate_observation()


def main():

    env = PipeWorld()

    while True:
        env.step()

        env.display()

        print(
            "What action with your budget",
            env.budget,
            " score: ",
            env.score,
            " Maintain (M), Inspect (I), Scale(S), None (N)",
        )
        action = input()
        if action == "S":
            scale = input()
            try:
                scale = float(scale)
                env.scale_segments(scale)
            except:
                pass
        elif action == "M" or action == "I":
            print("Which Logical Segment from 0-", len(env.logical_segments.logical_segments))
            index = input()
            try:
                index = int(index)
                if index >= 0 and index < len(env.logical_segments.logical_segments):
                    if action == "M":
                        env.maintain(index)
                    elif action == "I":
                        env.inspect(index)
            except:
                pass


if __name__ == "__main__":
    main()
