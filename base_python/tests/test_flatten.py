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

from data_pb2 import Space
from cogment_verse.spaces import flattened_dimensions


def test_flattened_dimensions_discrete():
    assert flattened_dimensions(Space(properties=[Space.Property(discrete=Space.Discrete(num=2))])) == 2

    assert (
        flattened_dimensions(
            Space(
                properties=[
                    Space.Property(
                        discrete=Space.Discrete(
                            labels=["brake", "accelerate", "do nothing"],
                            num=2,  # Will be ignored as there are more labels
                        )
                    )
                ]
            )
        )
        == 3
    )

    assert (
        flattened_dimensions(
            Space(
                properties=[
                    Space.Property(discrete=Space.Discrete(labels=["brake", "accelerate", "do nothing"], num=12))
                ]
            )
        )
        == 12
    )

    assert (
        flattened_dimensions(
            Space(
                properties=[
                    Space.Property(key="a", discrete=Space.Discrete(labels=["brake", "accelerate", "do nothing"])),
                    Space.Property(key="b", discrete=Space.Discrete(num=5)),
                ]
            )
        )
        == 8
    )


def test_flattened_dimensions_box():
    assert flattened_dimensions(Space(properties=[Space.Property(box=Space.Box(shape=[2]))])) == 2

    assert flattened_dimensions(Space(properties=[Space.Property(box=Space.Box(shape=[4]))])) == 4

    assert flattened_dimensions(Space(properties=[Space.Property(box=Space.Box(shape=[2, 3, 4]))])) == 24

    assert (
        flattened_dimensions(
            Space(
                properties=[
                    Space.Property(key="a", box=Space.Box(shape=[10])),
                    Space.Property(key="b", box=Space.Box(shape=[2, 3, 4])),
                ]
            )
        )
        == 34
    )


def test_flattened_dimensions_mixed():
    assert (
        flattened_dimensions(
            Space(
                properties=[
                    Space.Property(key="a", box=Space.Box(shape=[10])),
                    Space.Property(key="b", discrete=Space.Discrete(labels=["brake", "accelerate", "do nothing"])),
                    Space.Property(key="c", box=Space.Box(shape=[2, 3, 4])),
                ]
            )
        )
        == 37
    )
