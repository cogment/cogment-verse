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

import io

import torch
from cogment_verse_torch_agents.simple_a2c.simple_a2c_agent import SimpleA2CAgentAdapter
from data_pb2 import EnvironmentSpecs

# pylint: disable=protected-access


def test_create():
    adapter = SimpleA2CAgentAdapter()
    model, model_user_data = adapter._create(
        model_id="test",
        environment_specs=EnvironmentSpecs(implementation="foo", num_input=2, num_action=3),
        actor_network_hidden_size=10,
        critic_network_hidden_size=12,
    )
    assert model.model_id == "test"
    assert model.version_number == 1
    assert isinstance(model.actor_network, torch.nn.Module)
    assert isinstance(model.critic_network, torch.nn.Module)
    assert model_user_data == {"environment_implementation": "foo", "num_input": 2, "num_action": 3}


def test_save_and_load():
    adapter = SimpleA2CAgentAdapter()
    environment_specs = EnvironmentSpecs(implementation="foo", num_input=4, num_action=3)
    model, model_user_data = adapter._create(
        model_id="test",
        environment_specs=environment_specs,
    )

    with io.BytesIO() as serialized_model_io:
        adapter._save(
            model=model,
            model_user_data=model_user_data,
            model_data_f=serialized_model_io,
            environment_specs=environment_specs,
        )
        serialized_model_data = serialized_model_io.getvalue()

    deserialized_model = adapter._load(
        model_id="test",
        version_number=2,
        model_user_data=model_user_data,
        version_user_data={},
        model_data_f=io.BytesIO(serialized_model_data),
        environment_specs=environment_specs,
    )

    assert deserialized_model.model_id == "test"
    assert deserialized_model.version_number == 2
    assert isinstance(deserialized_model.actor_network, torch.nn.Module)
    for model_param, deserialized_model_param in zip(
        model.actor_network.parameters(), deserialized_model.actor_network.parameters()
    ):
        assert torch.equal(model_param, deserialized_model_param)
    assert isinstance(deserialized_model.critic_network, torch.nn.Module)
    for model_param, deserialized_model_param in zip(
        model.critic_network.parameters(), deserialized_model.critic_network.parameters()
    ):
        assert torch.equal(model_param, deserialized_model_param)
