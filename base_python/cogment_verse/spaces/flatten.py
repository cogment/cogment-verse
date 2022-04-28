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

# no import


def flattened_dimensions(space):
    """
    Computes the number of dimensions a flattened equivalent of the given space would have.
    """

    space_dim = 0
    for prop in space.properties:
        prop_dim = 0
        if prop.WhichOneof("definition") == "discrete":
            prop_dim = max(len(prop.discrete.labels), prop.discrete.num)
        else:
            # box
            if len(prop.box.shape) > 0:
                prop_dim = 1
                for dim in prop.box.shape:
                    prop_dim *= dim
        space_dim += prop_dim

    return space_dim
