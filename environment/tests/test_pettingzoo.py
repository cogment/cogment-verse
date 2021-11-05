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

from cogment_verse_environment.factory import make_environment

# pylint: disable=protected-access


def test_render():
    env = make_environment("pettingzoo", "connect_four_v3")
    env.reset()
    x = env.render()
    assert len(x.shape) == 3


def test_make_env():
    env = make_environment("pettingzoo", "connect_four_v3")
    env.reset()


def test_observation():
    env = make_environment("pettingzoo", "connect_four_v3")
    env.reset()
    obs = env.step(0)
    x = obs.observation
    assert x.shape == (84,)


def test_step():
    env = make_environment("pettingzoo", "connect_four_v3")
    env.reset()
    obs = env.step(0)
    x = obs.observation
    assert x.shape == (84,)
    # correct handling of np arrays
    _ = env.step(np.array(0))


def test_current_player():
    env = make_environment("pettingzoo", "connect_four_v3", num_players=2)
    obs = env.reset()
    assert env._num_players == 2
    assert obs.current_player == 0
    assert env._turn == 0
    obs = env.step(0)
    assert obs.current_player == 1
    obs = env.step(0)
    assert obs.current_player == 0


def test_rewards():
    env = make_environment("pettingzoo", "connect_four_v3", num_players=2)
    obs = env.reset()

    done = False
    cumulative_rewards = np.full(2, 0.0)

    while not done:
        action = np.random.choice(obs.legal_moves_as_int)
        obs = env.step(action)
        done = obs.done
        cumulative_rewards += obs.rewards

    assert cumulative_rewards.sum() == 0.0
