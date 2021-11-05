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

from cogment_verse_environment.factory import make_environment
from cogment_verse_torch_agents.wrapper import format_legal_moves
from cogment_verse_torch_agents.atari_cnn import NatureAtariDQNModel

import pytest
import numpy as np
from gym.envs import register

# pylint doesn't like test fixtures
# pylint: disable=redefined-outer-name

# Atari-py includes a free Tetris rom for testing without needing to download other ROMs
register(
    id="TetrisALENoFrameskip-v0",
    entry_point="gym.envs.atari:AtariEnv",
    kwargs={"game": "tetris", "obs_type": "image"},
    max_episode_steps=10000,
    nondeterministic=False,
)


@pytest.fixture
def act_dim():
    return 5


@pytest.fixture
def testing_env():
    return make_environment("atari", "TetrisALE")


def test_render(testing_env):
    obs = testing_env.reset()
    assert obs.current_player == 0
    assert obs.observation.shape == (84 * 84 * 4,)
    _ = testing_env.render()


def test_action(testing_env):
    _ = testing_env.reset()
    testing_env.step(0)


@pytest.mark.parametrize("framestack", [1, 2, 4])
def test_cnn_action(act_dim, framestack):
    env = make_environment("atari", "TetrisALE", flatten=False, framestack=framestack)
    obs = env.reset()
    print(obs.observation.shape)
    assert obs.observation.shape == (framestack, 84, 84)
    dqn = NatureAtariDQNModel(obs_dim=84 * 84, act_dim=act_dim, framestack=framestack)

    action = dqn.act(obs.observation, format_legal_moves(obs.legal_moves_as_int, act_dim))
    env.step(action)


def test_cnn_learn(act_dim):
    env = make_environment("atari", "TetrisALE", flatten=False, framestack=1)
    obs = env.reset()
    print(obs.observation.shape)
    dqn = NatureAtariDQNModel(obs_dim=84 * 84, act_dim=act_dim, framestack=1)

    legal_moves = format_legal_moves(obs.legal_moves_as_int, act_dim)
    _ = dqn.act(obs.observation, legal_moves)

    batch = {
        "observations": np.random.rand(4, 1, 84, 84).astype(np.float32),
        "next_observations": np.random.rand(4, 1, 84, 84).astype(np.float32),
        "rewards": np.random.rand(4, 1).astype(np.float32),
        "actions": np.random.randint(0, high=act_dim, size=4).reshape(-1, 1),
        "legal_moves_as_int": np.array([legal_moves] * 4),
        "done": np.random.randint(0, high=2, size=4).reshape(-1, 1),
    }

    info = dqn.learn(batch)
    assert "loss" in info


def test_replay_buffer(act_dim):
    env = make_environment("atari", "TetrisALE", flatten=False, framestack=1)
    obs = env.reset()
    print(obs.observation.shape)
    agent = NatureAtariDQNModel(obs_dim=84 * 84, act_dim=act_dim, framestack=1)

    for _ in range(32):
        legal_moves = format_legal_moves(obs.legal_moves_as_int, act_dim)
        action = agent.act(obs.observation, legal_moves)
        next_obs = env.step(action)

        data = (
            obs.observation.astype(np.float32),
            np.array(legal_moves),
            action,
            np.array(next_obs.rewards).astype(np.float32),
            next_obs.observation.astype(np.float32),
            np.array(legal_moves),
            next_obs.done,
        )

        agent.consume_training_sample(data)

        if next_obs.done:
            next_obs = env.reset()

        obs = next_obs

    assert agent.replay_buffer_size() == 32
    batch = agent.sample_training_batch(32)
    info = agent.learn(batch)
    assert "loss" in info
