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

import os
from tempfile import TemporaryDirectory

import pytest
from cogment_verse_torch_agents.third_party.td3.td3 import TD3Agent

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def obs_dim():
    return 8


@pytest.fixture
def act_dim():
    return 2


def test_serialize(obs_dim, act_dim):
    agent = TD3Agent(obs_dim=obs_dim, act_dim=act_dim)

    # pylint: disable=protected-access
    assert "min_action" in agent._params
    assert "max_action" in agent._params

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "agent.dat")

        agent.save(filepath)

        params = agent._params
        print("PARAMS", params.keys())

        agent2 = TD3Agent(obs_dim=obs_dim, act_dim=act_dim)
        agent2.load(filepath)
