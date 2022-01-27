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

import cogment.api.common_pb2 as common_api
from cogment_verse_torch_agents.selfplay_td3.wrapper import (
    tensor_from_cog_obs,
    tensor_from_cog_action,
    current_player_from_obs,
)

from collections import namedtuple


def get_agent_SARSD(sample, next_sample, last_tick):
    state = tensor_from_cog_obs(sample.get_actor_observation(1))
    next_state = tensor_from_cog_obs(next_sample.get_actor_observation(1))
    action = tensor_from_cog_action(sample.get_actor_action(1))
    reward = sample.get_actor_reward(1, default=0.0)
    agent = current_player_from_obs(sample.get_actor_observation(1))

    return (
        agent,
        state,
        action,
        reward,
        next_state,
        1 if last_tick else 0,
    )


TrainingSample = namedtuple(
    "TrainingSample",
    ["player_sample", "trial_cumulative_reward"],
)


async def sample_producer(run_sample_producer_session):
    previous_sample = None
    last_tick = False

    async for sample in run_sample_producer_session.get_all_samples():

        if sample.get_trial_state() == common_api.TrialState.ENDED:
            last_tick = True

        if previous_sample:
            # run_sample_producer_session.produce_training_sample(
            #     TrainingSample(
            #         player_sample=get_agent_SARSD(previous_sample, sample, last_tick),
            #         trial_cumulative_reward = 0,
            #     ),
            # )
            run_sample_producer_session.produce_training_sample(get_agent_SARSD(previous_sample, sample, last_tick))
        previous_sample = sample
