# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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

from cogment_verse import TorchReplayBuffer # pylint: disable=abstract-class-instantiated


def test_torch_replay_buffer():
    replay_buffer = TorchReplayBuffer(capacity=10, observation_shape=(2,), action_shape=(2, 2))
    assert replay_buffer.size() == 0
    assert replay_buffer.capacity == 10
    assert replay_buffer.num_total == 0

    replay_buffer.add(observation=[0, 1], next_observation=[2, 3], action=[[4, 5], [6, 7]], reward=12, done=1)

    assert replay_buffer.size() == 1
    assert replay_buffer.capacity == 10
    assert replay_buffer.num_total == 1

    sample = replay_buffer.sample(2)
    assert sample.size() == 1

    for i in range(100):
        replay_buffer.add(
            observation=[i, i + 1],
            next_observation=[i + 2, i + 3],
            action=[[i + 4, i + 5], [i + 6, i + 7]],
            reward=i + 8,
            done=0,
        )

    sample = replay_buffer.sample(10)
    assert sample.size() == 10
    assert sample.observation.shape == (10, 2)
    assert sample.next_observation.shape == (10, 2)
    assert sample.action.shape == (10, 2, 2)
    assert sample.reward.shape == (10,)
    assert sample.done.shape == (10,)
