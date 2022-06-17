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

import logging

import cogment
from cogment.api.common_pb2 import TrialState

from cogment_verse.specs import (
    AgentConfig,
    cog_settings,
    EnvironmentConfig,
    HUMAN_ACTOR_IMPL,
    OBSERVER_ACTOR_CLASS,
    PLAYER_ACTOR_CLASS,
    WEB_ACTOR_NAME,
)

log = logging.getLogger(__name__)


def extend_actor_config(actor_config_template, run_id, environment_specs):
    config = AgentConfig()
    if actor_config_template is not None:
        config.CopyFrom(actor_config_template)
    config.run_id = run_id
    # pylint: disable=no-member
    config.environment_specs.CopyFrom(environment_specs)
    return config


class PlayRun:
    default_cfg = {"num_trials": 1, "observer": False, "players": []}

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._environment_specs = environment_specs
        self._cfg = cfg

    async def total_rewards_producer_impl(self, sample_producer_session):
        actors_total_rewards = {
            actor_parameters.name: 0.0 for actor_parameters in sample_producer_session.trial_info.parameters.actors
        }
        num_ticks = 0
        async for sample in sample_producer_session.all_trial_samples():
            if sample.trial_state == TrialState.ENDED:
                break

            num_ticks += 1
            for actor_name, actor_sample in sample.actors_data.items():
                actors_total_rewards[actor_name] += actor_sample.reward if actor_sample.reward is not None else 0.0

        sample = (
            actors_total_rewards,
            num_ticks,
        )
        sample_producer_session.produce_sample(sample)

    async def impl(self, run_session):
        # We ignore additional actor configs
        if self._environment_specs.num_players > len(self._cfg.players):
            raise RuntimeError(
                f"Expecting at least {self._environment_specs.num_players} configured actors, got {len(self._cfg.players)}"
            )

        actors_params = []
        has_human_actor = False
        for actor_params in self._cfg.players[: self._environment_specs.num_players]:
            if actor_params.implementation == HUMAN_ACTOR_IMPL:
                if has_human_actor:
                    raise RuntimeError("Can't have more than one human involved in the trial")
                # Human actor
                actors_params.append(
                    cogment.ActorParameters(
                        cog_settings,
                        name=WEB_ACTOR_NAME,
                        class_name=PLAYER_ACTOR_CLASS,
                        implementation=HUMAN_ACTOR_IMPL,
                        config=extend_actor_config(
                            actor_config_template=actor_params.get("agent_config", None),
                            run_id=run_session.run_id,
                            environment_specs=self._environment_specs,
                        ),
                    )
                )
                has_human_actor = True
            else:
                actors_params.append(
                    cogment.ActorParameters(
                        cog_settings,
                        name=actor_params.name,
                        class_name=PLAYER_ACTOR_CLASS,
                        implementation=actor_params.implementation,
                        config=extend_actor_config(
                            actor_config_template=actor_params.get("agent_config", None),
                            run_id=run_session.run_id,
                            environment_specs=self._environment_specs,
                        ),
                    )
                )

        if self._cfg.observer:
            if has_human_actor:
                raise RuntimeError("Can't have more than one human involved in the trial")
            # Add an observer agent
            actors_params.append(
                cogment.ActorParameters(
                    cog_settings,
                    name=WEB_ACTOR_NAME,
                    class_name=OBSERVER_ACTOR_CLASS,
                    implementation=HUMAN_ACTOR_IMPL,
                    config=AgentConfig(
                        run_id=run_session.run_id,
                        environment_specs=self._environment_specs,
                    ),
                )
            )
            has_human_actor = True

        run_session.log_params(
            **{
                f"actor_{actor_idx}_implementation": actor_params.implementation
                for actor_idx, actor_params in enumerate(actors_params)
            },
            **{
                f"actor_{actor_idx}_model_id": actor_params.config.model_id
                for actor_idx, actor_params in enumerate(actors_params)
            },
            **{
                f"actor_{actor_idx}_model_version": actor_params.config.model_version
                for actor_idx, actor_params in enumerate(actors_params)
            },
            environment=self._environment_specs.implementation,
        )

        # Helper function to create a trial configuration
        trial_params = cogment.TrialParameters(
            cog_settings,
            environment_name="env",
            environment_implementation=self._environment_specs.implementation,
            environment_config=EnvironmentConfig(run_id=run_session.run_id, render=has_human_actor, seed=50),
            actors=actors_params,
        )

        # Rollout a bunch of trials
        for (_step_idx, _trial_id, _trial_idx, sample,) in run_session.start_and_await_trials(
            trials_id_and_params=[
                (f"{run_session.run_id}_{trial_idx}", trial_params) for trial_idx in range(self._cfg.num_trials)
            ],
            sample_producer_impl=self.total_rewards_producer_impl,
            num_parallel_trials=1,
        ):
            (actors_total_rewards, num_ticks) = sample
            run_session.log_metrics(
                **{
                    f"actor_{actor_name}_reward": actor_total_reward
                    for actor_name, actor_total_reward in actors_total_rewards.items()
                },
                total_reward=sum(actors_total_rewards.values()),
                num_ticks=num_ticks,
            )
        log.info("play ending")
