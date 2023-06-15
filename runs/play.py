# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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
from google.protobuf.json_format import ParseDict
from cogment_verse.constants import ActorSpecType

from cogment_verse.specs import (
    HUMAN_ACTOR_IMPL,
    OBSERVER_ACTOR_CLASS,
    PLAYER_ACTOR_CLASS,
    WEB_ACTOR_NAME,
    AgentConfig,
    EnvironmentConfig,
    cog_settings,
)

log = logging.getLogger(__name__)


def extend_actor_config(actor_config_template, run_id, environment_specs, spec_type, seed):
    config = AgentConfig()
    if actor_config_template is not None:
        ParseDict(actor_config_template, config)
    config.run_id = run_id
    # pylint: disable=no-member
    config.environment_specs.CopyFrom(environment_specs)
    config.spec_type = spec_type
    config.seed = seed
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
                f"Expecting at least {self._environment_specs.num_players} configured actors, got {len(self._cfg.players)}. Verify the experiment and environment configuration files. Players registered: [{', '.join([player['name'] for player in self._cfg.players])}]"
            )

        players_cfg = self._cfg.players[: self._environment_specs.num_players]

        print(f"self._cfg.players: {self._cfg.players}")
        print(f"players_cfg: {players_cfg}")

        run_session.log_params(
            **{
                f"actor_{actor_idx}_implementation": actor_params.implementation
                for actor_idx, actor_params in enumerate(players_cfg)
            },
            **{
                f"actor_{actor_idx}_model_id": actor_params.get("config", {"model_id": None})["model_id"]
                for actor_idx, actor_params in enumerate(players_cfg)
            },
            **{
                f"actor_{actor_idx}_model_iteration": actor_params.get("config", {"model_iteration": None})[
                    "model_iteration"
                ]
                for actor_idx, actor_params in enumerate(players_cfg)
            },
            environment=self._environment_specs.implementation,
        )

        def create_trial_params(trial_idx):
            actors_params = []
            has_human_actor = False
            for actor_idx, actor_params in enumerate(players_cfg):
                if actor_params.implementation == HUMAN_ACTOR_IMPL:
                    if has_human_actor:
                        raise RuntimeError("Can't have more than one human involved in the trial")
                    # Human actor
                    spec_type = actor_params.get("spec_type", ActorSpecType.DEFAULT.value)
                    actors_params.append(
                        cogment.ActorParameters(
                            cog_settings,
                            name=WEB_ACTOR_NAME,
                            class_name=spec_type,
                            implementation=HUMAN_ACTOR_IMPL,
                            config=extend_actor_config(
                                actor_config_template=actor_params.get("agent_config", None),
                                run_id=run_session.run_id,
                                environment_specs=self._environment_specs.serialize(),
                                spec_type=spec_type,
                                seed=self._cfg.seed * (trial_idx + 1) * (actor_idx + 1),
                            ),
                        )
                    )
                    has_human_actor = True
                else:
                    spec_type = actor_params.get("spec_type", ActorSpecType.DEFAULT.value)
                    actors_params.append(
                        cogment.ActorParameters(
                            cog_settings,
                            name=actor_params.name,
                            class_name=spec_type,
                            implementation=actor_params.implementation,
                            config=extend_actor_config(
                                actor_config_template=actor_params.get("agent_config", None),
                                run_id=run_session.run_id,
                                environment_specs=self._environment_specs.serialize(),
                                spec_type=spec_type,
                                seed=self._cfg.seed * (trial_idx + 1) * (actor_idx + 1),
                            ),
                        )
                    )

            if self._cfg.observer:
                if has_human_actor:
                    raise RuntimeError("Can't have more than one human involved in the trial")
                # Add an observer agent
                spec_type = actor_params.get("spec_type", OBSERVER_ACTOR_CLASS)
                actors_params.append(
                    cogment.ActorParameters(
                        cog_settings,
                        name=WEB_ACTOR_NAME,
                        class_name=OBSERVER_ACTOR_CLASS,
                        implementation=HUMAN_ACTOR_IMPL,
                        config=AgentConfig(
                            run_id=run_session.run_id,
                            environment_specs=self._environment_specs.serialize(),
                            spec_type=spec_type,
                            seed=(self._cfg.seed + 100) * trial_idx,
                        ),
                    )
                )
                has_human_actor = True

            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id, render=has_human_actor, seed=self._cfg.seed * trial_idx
                ),
                actors=actors_params,
            )

        # Rollout a bunch of trials
        for (_step_idx, _trial_id, _trial_idx, sample,) in run_session.start_and_await_trials(
            trials_id_and_params=[
                (f"{run_session.run_id}_{trial_idx}", create_trial_params(trial_idx))
                for trial_idx in range(self._cfg.num_trials)
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
