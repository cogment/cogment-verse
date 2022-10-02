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

from .value import create_space_values
from .ndarray import serialize_ndarray


def sample_space(space, num_samples=1, rng=None, mask=None):
    if rng is None:
        rng = np.random.default_rng()

    space_values = create_space_values(space, num_samples)

    for prop_idx, prop in enumerate(space.properties):
        if prop.WhichOneof("type") == "discrete":
            if mask is not None and prop_idx < len(mask.properties) and len(mask.properties[prop_idx].discrete) > 0:
                for space_value, sampled_value in zip(
                    space_values, rng.choice(a=mask.properties[prop_idx].discrete, size=num_samples)
                ):
                    space_value.properties[prop_idx].discrete = sampled_value
                continue
            num_action = max(len(prop.discrete.labels), prop.discrete.num)
            for space_value, sampled_value in zip(space_values, rng.integers(low=0, high=num_action, size=num_samples)):
                space_value.properties[prop_idx].discrete = sampled_value
        else:
            shape = prop.box.shape

            high = np.array([bound.bound if bound.HasField("bound") else np.inf for bound in prop.box.high]).reshape(
                shape
            )
            low = np.array([bound.bound if bound.HasField("bound") else -np.inf for bound in prop.box.low]).reshape(
                shape
            )

            sampled_values = np.zeros((num_samples, *shape))

            if len(shape) == 1:
                high = np.repeat(high[np.newaxis, :], num_samples, axis=0)
                low = np.repeat(low[np.newaxis, :], num_samples, axis=0)
            else:
                high = np.repeat(high[np.newaxis, :, :], num_samples, axis=0)
                low = np.repeat(low[np.newaxis, :, :], num_samples, axis=0)

            unbounded_mask = (high == np.inf) & (low == -np.inf)
            sampled_values[unbounded_mask] = rng.normal(size=unbounded_mask[unbounded_mask].shape)

            low_bounded_mask = (high == np.inf) & (low != -np.inf)
            sampled_values[low_bounded_mask] = (
                -rng.exponential(size=low_bounded_mask[low_bounded_mask].shape) + low[low_bounded_mask]
            )

            high_bounded_mask = (high != np.inf) & (low == -np.inf)
            sampled_values[high_bounded_mask] = (
                rng.exponential(size=high_bounded_mask[high_bounded_mask].shape) + high[high_bounded_mask]
            )

            bounded_mask = (high != np.inf) & (low != -np.inf)
            sampled_values[bounded_mask] = rng.uniform(
                low=low[bounded_mask], high=high[bounded_mask], size=bounded_mask[bounded_mask].shape
            )

            for space_value, sampled_value in zip(space_values, sampled_values):
                space_value.properties[prop_idx].box.CopyFrom(serialize_ndarray(sampled_value))

    return space_values
