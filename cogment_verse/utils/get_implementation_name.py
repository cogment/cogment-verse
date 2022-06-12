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

# no import


def get_implementation_name(instance):
    base_impl_name = f"{type(instance).__module__}.{type(instance).__name__}"
    if hasattr(instance, "get_implementation_name"):
        instance_impl_name = instance.get_implementation_name()
        if instance_impl_name is not None and instance_impl_name != "":
            return f"{base_impl_name}/{instance_impl_name}"
    return base_impl_name
