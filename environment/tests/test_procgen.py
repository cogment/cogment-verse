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
from cogment_verse_environment.procgen_env import ENV_NAMES, ProcGenEnv
from cogment_verse_environment.utils.serialization_helpers import deserialize_img, deserialize_np_array
from data_pb2 import AgentAction, EnvironmentConfig
from mock_environment_session import ActorInfo

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name


@pytest.mark.asyncio
@pytest.mark.parametrize("framestack", [1, 2, 4])
@pytest.mark.parametrize("env_name", ENV_NAMES)
@pytest.mark.parametrize("flatten", [True, False])
async def test_observation(create_mock_environment_session, env_name, framestack, flatten):
    session = create_mock_environment_session(
        impl_name=f"procgen/{env_name}",
        trial_id="test_procgen",
        environment_config=EnvironmentConfig(framestack=framestack, flatten=flatten),
        actor_infos=[ActorInfo("player_1", "player")],
    )

    tick_0_events = await session.receive_events()
    assert tick_0_events.tick_id == 0
    assert len(tick_0_events.rewards) == 0
    assert len(tick_0_events.messages) == 0
    assert len(tick_0_events.observations) == 1

    tick_0_observation_destination, tick_0_observation = tick_0_events.observations[0]
    assert tick_0_observation_destination == "*"
    if flatten:
        assert deserialize_np_array(tick_0_observation.vectorized).shape == (64 * 64 * framestack,)
    else:
        assert deserialize_np_array(tick_0_observation.vectorized).shape == (framestack, 64, 64)

    assert deserialize_img(tick_0_observation.pixel_data).shape == (1, 1, 3)

    await session.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("env_name", ENV_NAMES)
async def test_step(create_mock_environment_session, env_name):
    session = create_mock_environment_session(
        impl_name=f"procgen/{env_name}",
        trial_id="test_procgen",
        environment_config=EnvironmentConfig(framestack=4, flatten=True),
        actor_infos=[ActorInfo("player_1", "player")],
    )
    tick_0_events = await session.receive_events()
    assert tick_0_events.tick_id == 0

    session.send_events(actions=[AgentAction(discrete_action=0)])

    tick_1_events = await session.receive_events()
    assert tick_1_events.tick_id == 1
    await session.terminate()


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_render(env_name):
    env = ProcGenEnv(env_name=env_name, flatten=False)
    env.reset()
    for _ in range(10):
        action = np.random.randint(0, 15)
        obs = env.step(action)
        pixels = env.render()
        assert pixels.shape == (64, 64, 3)
        assert np.allclose(obs.observation[0], np.mean(pixels, axis=2).astype(np.uint8))
