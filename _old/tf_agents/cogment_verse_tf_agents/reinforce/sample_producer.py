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

from collections import namedtuple

import cogment.api.common_pb2 as common_api
from cogment_verse_tf_agents.wrapper import tf_action_from_cog_action, tf_obs_from_cog_obs


def vectorized_training_sample_from_samples(sample, next_sample, last_tick):
    vectorized_observation = tf_obs_from_cog_obs(sample.get_actor_observation(0))
    vectorized_next_observation = tf_obs_from_cog_obs(next_sample.get_actor_observation(0))
    action = tf_action_from_cog_action(sample.get_actor_action(0))
    reward = sample.get_actor_reward(0, default=0.0)

    return (
        vectorized_observation["vectorized"],
        action,
        reward,
        vectorized_next_observation["vectorized"],
        1 if last_tick else 0,
    )


TrainingSample = namedtuple(
    "TrainingSample",
    ["player_sample", "trial_cumulative_reward"],
)


async def sample_producer(run_sample_producer_session):
    num_actors = run_sample_producer_session.count_actors()
    previous_sample = None
    trial_cumulative_reward = 0
    last_tick = False

    async for sample in run_sample_producer_session.get_all_samples():

        if sample.get_trial_state() == common_api.TrialState.ENDED:
            last_tick = True

        trial_cumulative_reward += sum(
            [sample.get_actor_reward(actor_idx, default=0.0) for actor_idx in range(num_actors)]
        )

        if previous_sample:
            run_sample_producer_session.produce_training_sample(
                TrainingSample(
                    player_sample=vectorized_training_sample_from_samples(previous_sample, sample, last_tick),
                    trial_cumulative_reward=trial_cumulative_reward,
                ),
            )

        previous_sample = sample
