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
from cogment_verse_torch_agents.wrapper import format_legal_moves, torch_action_from_cog_action, torch_obs_from_cog_obs


def vectorized_training_sample_from_samples(
    sample, next_sample, last_tick, num_action, reward_override=None, actor_idx=None
):
    if actor_idx:
        current_player_actor_idx = actor_idx
    else:
        # Retrieve the current player's actor_idx from any actor's observation
        curr_obs = sample.get_actor_observation(0)
        next_obs = next_sample.get_actor_observation(0)

        # if player_override is set, it means that the previous action came from the teacher/expert
        # instead of the true current player
        if next_obs.player_override != -1:
            current_player_actor_idx = next_obs.player_override
        else:
            current_player_actor_idx = curr_obs.current_player

    vectorized_observation = torch_obs_from_cog_obs(sample.get_actor_observation(current_player_actor_idx))

    vectorized_next_observation = torch_obs_from_cog_obs(next_sample.get_actor_observation(current_player_actor_idx))

    action = sample.get_actor_action(current_player_actor_idx)

    if reward_override:
        reward = reward_override
    else:
        reward = sample.get_actor_reward(current_player_actor_idx, default=0.0)

    return (
        vectorized_observation["vectorized"],
        format_legal_moves(vectorized_observation["legal_moves_as_int"], num_action),
        torch_action_from_cog_action(action),
        reward,
        vectorized_next_observation["vectorized"],
        format_legal_moves(vectorized_next_observation["legal_moves_as_int"], num_action),
        1 if last_tick else 0,
    )


TrainingSample = namedtuple(
    "TrainingSample",
    ["current_player_sample", "trial_total_reward"],
)


async def sample_producer(run_sample_producer_session):
    num_actors = run_sample_producer_session.count_actors()

    if not run_sample_producer_session.run_config.aggregate_by_actor:
        previous_sample = None

        trial_total_reward = 0

        last_tick = False

        async for sample in run_sample_producer_session.get_all_samples():

            if sample.get_trial_state() == common_api.TrialState.ENDED:
                last_tick = True

            trial_total_reward += sum(
                [sample.get_actor_reward(actor_idx, default=0.0) for actor_idx in range(num_actors)]
            )

            if previous_sample:
                run_sample_producer_session.produce_training_sample(
                    TrainingSample(
                        current_player_sample=vectorized_training_sample_from_samples(
                            previous_sample,
                            sample,
                            last_tick,
                            run_sample_producer_session.run_config.environment.specs.num_action,
                        ),
                        trial_total_reward=trial_total_reward if last_tick else None,
                    ),
                )

            previous_sample = sample
    else:
        # todo: the logic below is incorrect when there is human/expert intervention
        # and needs to be modified to support HILL with cooperative multiplayer games
        distinguished_actor = run_sample_producer_session.get_trial_config().distinguished_actor

        previous_samples = [None] * num_actors
        actor_rewards = [0.0] * num_actors
        actor_cumulative_rewards = [0.0] * num_actors

        trial_total_reward = 0

        last_tick = False
        current_player = 0

        async for sample in run_sample_producer_session.get_all_samples():
            assert current_player == sample.get_actor_observation(0).current_player

            if sample.get_trial_state() == common_api.TrialState.ENDED:
                last_tick = True

            # update the rewards for the current turn
            for actor_idx in range(num_actors):
                reward = sample.get_actor_reward(actor_idx, default=0.0)
                actor_rewards[actor_idx] += reward
                actor_cumulative_rewards[actor_idx] += reward

                if distinguished_actor in (-1, actor_idx):
                    trial_total_reward += reward

            for actor_idx in range(num_actors):
                if actor_idx == current_player or last_tick:
                    if previous_samples[actor_idx]:
                        run_sample_producer_session.produce_training_sample(
                            TrainingSample(
                                current_player_sample=vectorized_training_sample_from_samples(
                                    previous_samples[actor_idx],
                                    sample,
                                    last_tick,
                                    run_sample_producer_session.run_config.num_action,
                                    reward_override=actor_rewards[actor_idx],
                                    actor_idx=actor_idx,
                                ),
                                trial_total_reward=trial_total_reward if last_tick else None,
                            ),
                        )
                        actor_rewards[actor_idx] = 0.0

            previous_samples[current_player] = sample
            current_player = (current_player + 1) % num_actors
