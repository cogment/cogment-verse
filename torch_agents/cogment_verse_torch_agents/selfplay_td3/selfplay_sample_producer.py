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
    tensor_from_cog_goal,
)


def get_SARSDG(sample, next_sample, last_tick, num_trials):
    SARSDG = []
    for trial_num in range(num_trials):
        SARSDG.append((int(current_player_from_obs(sample.get_actor_observation(trial_num))), # TBD: function names
                        tensor_from_cog_obs(sample.get_actor_observation(trial_num)),
                        tensor_from_cog_action(sample.get_actor_action(trial_num)),
                        sample.get_actor_reward(trial_num, default=0.0),
                        tensor_from_cog_obs(next_sample.get_actor_observation(trial_num)),
                        1 if last_tick else 0,
                        tensor_from_cog_goal(sample.get_actor_observation(trial_num)),
                          ))
    return SARSDG


async def sample_producer(run_sample_producer_session):
    num_trials = run_sample_producer_session.count_actors()
    previous_sample = None
    last_tick = False

    async for sample in run_sample_producer_session.get_all_samples():

        if sample.get_trial_state() == common_api.TrialState.ENDED:
            last_tick = True

        if previous_sample:
            run_sample_producer_session.produce_training_sample(get_SARSDG(previous_sample, sample, last_tick, num_trials))
        previous_sample = sample
