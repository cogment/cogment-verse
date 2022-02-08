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

import pytest
from cogment_verse_environment.environment_adapter import EnvironmentAdapter
from mock_environment_session import MockEnvironmentSession

# pylint: disable=protected-access


@pytest.fixture
def create_mock_environment_session():
    adapter = EnvironmentAdapter()
    implementations = adapter._create_implementations()

    def _create(impl_name, trial_id, environment_config, actor_infos):
        return MockEnvironmentSession(
            trial_id=f"test_pettingzoo/{trial_id}",
            environment_config=environment_config,
            actor_infos=actor_infos,
            environment_impl=implementations[impl_name],
        )

    return _create
