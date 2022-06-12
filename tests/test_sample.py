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

from cogment_verse.specs import (
    Space,
    sample_space,
    deserialize_ndarray,
)


def test_sample_discrete():
    space = Space(properties=[Space.Property(discrete=Space.Discrete(num=2))])

    values = sample_space(space, 4)
    assert len(values) == 4

    for value in values:
        assert len(value.properties) == 1
        assert value.properties[0].discrete in range(2)


def test_sample_discrete_named():
    space = Space(properties=[Space.Property(discrete=Space.Discrete(labels=["brake", "accelerate", "do nothing"]))])

    values = sample_space(space, 12)
    assert len(values) == 12

    for value in values:
        assert len(value.properties) == 1
        assert value.properties[0].discrete in range(3)


def test_sample_discrete_named_and_more():
    space = Space(
        properties=[Space.Property(discrete=Space.Discrete(labels=["brake", "accelerate", "do nothing"], num=12))]
    )

    values = sample_space(space, 12)
    assert len(values) == 12

    for value in values:
        assert len(value.properties) == 1
        assert value.properties[0].discrete in range(12)


def test_sample_box():
    space = Space(
        properties=[
            Space.Property(
                box=Space.Box(
                    shape=[2, 3],
                    low=[
                        Space.Bound(bound=-1),
                        Space.Bound(bound=-2),
                        Space.Bound(),
                        Space.Bound(bound=-4),
                        Space.Bound(),
                        Space.Bound(bound=-6),
                    ],
                    high=[
                        Space.Bound(bound=1),
                        Space.Bound(bound=2),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(bound=5),
                        Space.Bound(bound=6),
                    ],
                )
            )
        ]
    )

    values = sample_space(space, 4)
    assert len(values) == 4

    for value in values:
        assert len(value.properties) == 1
        assert value.properties[0].box.shape == [2, 3]
        ndarray = deserialize_ndarray(value.properties[0].box)
        assert ndarray.shape == (2, 3)
