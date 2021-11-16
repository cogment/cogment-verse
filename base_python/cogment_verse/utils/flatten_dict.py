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

from collections import OrderedDict


def flatten_dict(nested_dict, prefix="", delimiter="/"):
    flat_dict = {}
    if prefix and prefix[-1] != delimiter:
        prefix = prefix + delimiter

    for key, val in nested_dict.items():
        if isinstance(val, (dict, OrderedDict)):
            child_dict = flatten_dict(val, f"{prefix}{key}{delimiter}")
            for child_key, child_val in child_dict.items():
                flat_dict[child_key] = child_val
        else:
            flat_dict[f"{prefix}{key}"] = val

    return flat_dict
