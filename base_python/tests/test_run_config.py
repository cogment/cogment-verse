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
import yaml
from data_pb2 import RunConfig
from google.protobuf.json_format import MessageToDict, ParseDict

# for pytest fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def config_dict():
    config_str = """
environment:
    specs:
        implementation: gym/CartPole-v0
        num_players: 1
        num_input: 4
        num_action: 2
    config:
        render_width: 256
        flatten: True
        framestack: 1
epsilon_min: 0.1
epsilon_steps: 100000
target_net_update_schedule: 1000
learning_rate: 1.0e-4
lr_warmup_steps: 10000
demonstration_count: 0
total_trial_count: 10000
model_publication_interval: 1000
model_archive_interval: 4000 # Archive every 4000 training steps
batch_size: 256
min_replay_buffer_size: 1000
max_parallel_trials: 4
model_kwargs: {}
max_replay_buffer_size: 100000
aggregate_by_actor: False
replay_buffer_config:
    observation_dtype: float32
    action_dtype: int8
agent_implementation: rainbowtorch
"""
    return yaml.safe_load(config_str)


def test_config(config_dict):
    # should not raise exception
    run_config = ParseDict(config_dict, RunConfig())
    dct = MessageToDict(run_config, preserving_proto_field_name=True)
    assert "environment" in dct
    assert "replay_buffer_config" in dct
    assert "observation_dtype" in dct["replay_buffer_config"]
