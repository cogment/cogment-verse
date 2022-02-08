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
from cogment_verse_environment.utils.serialization_helpers import deserialize_img, deserialize_np_array
from data_pb2 import AgentAction, EnvironmentConfig
from mock_environment_session import ActorInfo

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.mark.asyncio
@pytest.mark.parametrize("framestack", [1, 2, 4])
async def test_observation(create_mock_environment_session, framestack):
    tetris_session = create_mock_environment_session(
        impl_name="atari/TetrisALE",
        trial_id="test_atari",
        environment_config=EnvironmentConfig(framestack=framestack, flatten=True),
        actor_infos=[ActorInfo("player_1", "player")],
    )

    tick_0_events = await tetris_session.receive_events()
    assert tick_0_events.tick_id == 0
    assert len(tick_0_events.rewards) == 0
    assert len(tick_0_events.messages) == 0
    assert len(tick_0_events.observations) == 1

    tick_0_observation_destination, tick_0_observation = tick_0_events.observations[0]
    assert tick_0_observation_destination == "*"
    assert deserialize_np_array(tick_0_observation.vectorized).shape == (84 * 84 * framestack,)
    assert deserialize_img(tick_0_observation.pixel_data).shape == (1, 1, 3)

    await tetris_session.terminate()


@pytest.fixture
@pytest.mark.asyncio
async def tetris_session(create_mock_environment_session):
    session = create_mock_environment_session(
        impl_name="atari/TetrisALE",
        trial_id="test_atari",
        environment_config=EnvironmentConfig(framestack=4, flatten=True),
        actor_infos=[ActorInfo("player_1", "player")],
    )

    yield session
    await session.terminate()


@pytest.mark.asyncio
async def test_step(tetris_session):
    tick_0_events = await tetris_session.receive_events()
    assert tick_0_events.tick_id == 0

    tetris_session.send_events(actions=[AgentAction(discrete_action=0)])

    tick_1_events = await tetris_session.receive_events()
    assert tick_1_events.tick_id == 1
