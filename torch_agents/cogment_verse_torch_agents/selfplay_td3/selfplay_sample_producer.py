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
    tensor_from_cog_state,
    tensor_from_cog_img,
    tensor_from_cog_action,
    current_player_from_obs,
    tensor_from_cog_goal,
    current_player_done_flag,
    trial_done_flag,
)
from collections import namedtuple

Sample = namedtuple("Sample", ["current_player", "state", "img",
                                "action", "reward", "next_state",
                                "next_img", "player_done", "trial_done",
                                "goal"])


def get_samples(sample, next_sample):
    sample_player_done = current_player_done_flag(sample.get_actor_observation(0)),
    next_sample_player_done = current_player_done_flag(next_sample.get_actor_observation(0)),

    if not (sample_player_done and not next_sample_player_done):
        current_player = int(current_player_from_obs(sample.get_actor_observation(0)))
        return Sample(
            current_player=current_player, # TBD: function names
            state=tensor_from_cog_state(sample.get_actor_observation(current_player)),
            img=tensor_from_cog_img(sample.get_actor_observation(current_player)),
            action=tensor_from_cog_action(sample.get_actor_action(current_player)),
            reward=sample.get_actor_reward(current_player, default=0.0),
            next_state=tensor_from_cog_state(next_sample.get_actor_observation(current_player)),
            next_img=tensor_from_cog_img(next_sample.get_actor_observation(current_player)),
            player_done=current_player_done_flag(next_sample.get_actor_observation(current_player)),
            trial_done=1 if next_sample.get_trial_state() == common_api.TrialState.ENDED else 0,  # trial end flag never set,
            goal=tensor_from_cog_goal(sample.get_actor_observation(current_player)),
                          )
    else:
        return ()


async def sample_producer(run_sample_producer_session):
    previous_sample = None

    async for sample in run_sample_producer_session.get_all_samples():
        if previous_sample:
            Sample = get_samples(previous_sample, sample)
            if Sample:
                run_sample_producer_session.produce_training_sample(Sample)
        previous_sample = sample
