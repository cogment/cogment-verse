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

import numpy as np

from cogment_verse.specs import Space, SpaceMask, sample_space, flattened_dimensions, flatten, flatten_mask, unflatten


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


def test_flatten():
    space = Space(
        properties=[
            Space.Property(key="a", discrete=Space.Discrete(labels=["brake", "accelerate", "do nothing"])),
            Space.Property(
                key="b",
                box=Space.Box(
                    shape=[2, 3],
                    low=[
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                    ],
                    high=[
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                    ],
                ),
            ),
        ]
    )
    [value] = sample_space(space)
    flat_value = flatten(space, value)
    assert flat_value.shape == (9,)
    unflattened_value = unflatten(space, flat_value)
    assert unflattened_value == value

    unflattened_random_value = unflatten(space, np.random.default_rng(64).normal(size=(9,)))
    assert len(unflattened_random_value.properties) == 2
    assert unflattened_random_value.properties[0].discrete in [0, 1, 2]
    assert unflattened_random_value.properties[0].box is not None


def test_flatten_mask():
    space = Space(
        properties=[
            Space.Property(
                key="a",
                box=Space.Box(
                    shape=[2, 3],
                    low=[
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                    ],
                    high=[
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                        Space.Bound(),
                    ],
                ),
            ),
            Space.Property(key="b", discrete=Space.Discrete(labels=["brake", "accelerate", "do nothing"])),
        ]
    )
    mask = SpaceMask(properties=[SpaceMask.PropertyMask(), SpaceMask.PropertyMask(discrete=[1, 2])])
    flat_mask = flatten_mask(space, mask)
    assert flat_mask.shape == (9,)
    assert np.array_equal(flat_mask, np.array([1, 1, 1, 1, 1, 1, 0, 1, 1]))
