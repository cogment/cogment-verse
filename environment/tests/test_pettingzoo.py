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

import numpy as np
import pytest
from cogment_verse_environment.utils.serialization_helpers import deserialize_img, deserialize_np_array
from data_pb2 import AgentAction, EnvironmentConfig
from mock_environment_session import ActorInfo

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
@pytest.mark.asyncio
async def connect_four_session(create_mock_environment_session):
    session = create_mock_environment_session(
        impl_name="pettingzoo/connect_four_v3",
        trial_id="test_pettingzoo",
        environment_config=EnvironmentConfig(framestack=1, flatten=True),
        actor_infos=[ActorInfo("player_1", "player"), ActorInfo("player_2", "player")],
    )

    yield session
    await session.terminate()


@pytest.mark.asyncio
async def test_observation(connect_four_session):
    tick_0_events = await connect_four_session.receive_events()
    assert tick_0_events.tick_id == 0
    assert len(tick_0_events.rewards) == 0
    assert len(tick_0_events.messages) == 0
    assert len(tick_0_events.observations) == 1

    tick_0_observation_destination, tick_0_observation = tick_0_events.observations[0]
    assert tick_0_observation_destination == "*"
    assert deserialize_np_array(tick_0_observation.vectorized).shape == (84,)
    assert deserialize_img(tick_0_observation.pixel_data).shape == (1, 1, 3)


@pytest.mark.asyncio
async def test_step(connect_four_session):
    tick_0_events = await connect_four_session.receive_events()
    assert tick_0_events.tick_id == 0

    connect_four_session.send_events(actions=[AgentAction(discrete_action=0), AgentAction(discrete_action=0)])

    tick_1_events = await connect_four_session.receive_events()
    assert tick_1_events.tick_id == 1


@pytest.mark.asyncio
async def test_current_player(connect_four_session):
    tick_0_events = await connect_four_session.receive_events()
    assert tick_0_events.tick_id == 0

    _, tick_0_observation = tick_0_events.observations[0]
    assert tick_0_observation.current_player == 0

    connect_four_session.send_events(actions=[AgentAction(discrete_action=0), AgentAction(discrete_action=0)])

    tick_1_events = await connect_four_session.receive_events()
    assert tick_1_events.tick_id == 1

    _, tick_1_observation = tick_1_events.observations[0]
    assert tick_1_observation.current_player == 1

    connect_four_session.send_events(actions=[AgentAction(discrete_action=0), AgentAction(discrete_action=0)])

    tick_2_events = await connect_four_session.receive_events()
    assert tick_2_events.tick_id == 2

    _, tick_2_observation = tick_2_events.observations[0]
    assert tick_2_observation.current_player == 0


@pytest.mark.asyncio
async def test_rewards(connect_four_session):
    cumulative_rewards = 0

    while True:
        tick_events = await connect_four_session.receive_events()
        if tick_events.done:
            break
        cumulative_rewards += sum([reward.value for reward in tick_events.rewards])
        _, observation = tick_events.observations[0]

        actions = [AgentAction(discrete_action=0), AgentAction(discrete_action=0)]
        actions[observation.current_player] = AgentAction(
            discrete_action=np.random.choice(observation.legal_moves_as_int)
        )

        connect_four_session.send_events(actions=actions)

    assert cumulative_rewards == 0
