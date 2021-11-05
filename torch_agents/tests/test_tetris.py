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

# import numpy as np
# import pytest

# from cogment_verse_torch_agents.wrapper import format_legal_moves
# from cogment_verse_torch_agents.env.factory import make_environment
# from cogment_verse_torch_agents.atari_cnn import NatureAtariDQNModel

# # pylint doesn't like test fixtures
# # pylint: disable=redefined-outer-name


# @pytest.fixture
# def act_dim():
#     return 12


# @pytest.fixture
# def env():
#     return make_environment("tetris", "TetrisA-v0")


# @pytest.fixture
# def env_nonflat():
#     return make_environment("tetris", "TetrisA-v0", flatten=False)


# def test_render(env):
#     obs = env.reset()
#     assert obs.current_player == 0
#     assert obs.observation.shape[:2] == (84 * 84 * 4,)
#     _ = env.render()


# def test_action(env):
#     _ = env.reset()
#     env.step(0)


# def test_cnn_action(env_nonflat, act_dim):
#     obs = env_nonflat.reset()
#     dqn = NatureAtariDQNModel(obs_dim=84 * 84, act_dim=act_dim, framestack=1)
#     action = dqn.act(obs.observation, format_legal_moves(obs.legal_moves_as_int, act_dim))
#     env_nonflat.step(action)


# def test_cnn_learn(env, act_dim):
#     obs = env.reset()
#     dqn = NatureAtariDQNModel(obs_dim=84 * 84, act_dim=act_dim)

#     legal_moves = format_legal_moves(obs.legal_moves_as_int, act_dim)
#     _ = dqn.act(obs.observation, legal_moves)

#     batch = {
#         "observations": np.random.rand(4, 4, 84, 84).astype(np.float32),
#         "next_observations": np.random.rand(4, 4, 84, 84).astype(np.float32),
#         "rewards": np.random.rand(4, 1).astype(np.float32),
#         "actions": np.random.randint(0, high=act_dim, size=4).reshape(-1, 1),
#         "legal_moves_as_int": np.array([legal_moves] * 4),
#         "done": np.random.randint(0, high=2, size=4).reshape(-1, 1),
#     }

#     info = dqn.learn(batch)
#     assert "loss" in info
