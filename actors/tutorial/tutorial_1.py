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

from cogment_verse.specs import (
    HUMAN_ACTOR_IMPL,
    PLAYER_ACTOR_CLASS,
    TEACHER_ACTOR_CLASS,
    WEB_ACTOR_NAME,
    AgentConfig,
    EnvironmentConfig,
    EnvironmentSpecs,
    cog_settings,
)

log = logging.getLogger(__name__)


class SimpleBCActor:
    def __init__(self, _cfg):
        super().__init__()

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config

        environment_specs = EnvironmentSpecs.deserialize(config.environment_specs)
        action_space = environment_specs.get_action_space(seed=config.seed)

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                action = action_space.sample()
                actor_session.do_action(action_space.serialize(action))


class SimpleBCTraining:
    default_cfg = {
        "seed": 12,
        "num_trials": 10,
    }

    def __init__(self, environment_specs, cfg):
        super().__init__()
        self._environment_specs = environment_specs
        self._cfg = cfg

    async def sample_producer(self, sample_producer_session):
        assert len(sample_producer_session.trial_info.parameters.actors) == 2

        players_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == PLAYER_ACTOR_CLASS
        ]
        teachers_params = [
            actor_params
            for actor_params in sample_producer_session.trial_info.parameters.actors
            if actor_params.class_name == TEACHER_ACTOR_CLASS
        ]
        assert len(players_params) == 1
        assert len(teachers_params) == 1
        player_params = players_params[0]
        teacher_params = teachers_params[0]

        environment_specs = EnvironmentSpecs.deserialize(player_params.config.environment_specs)
        action_space = environment_specs.get_action_space()

        async for sample in sample_producer_session.all_trial_samples():
            teacher_action = action_space.deserialize(sample.actors_data[teacher_params.name].action)

            if teacher_action.flat_value is not None:
                log.info(f"Got raw sample with action override from [{teacher_params.name}]")
            else:
                log.info(f"Got raw sample with action from [{player_params.name}]")

    async def impl(self, run_session):
        assert self._environment_specs.num_players == 1

        run_session.log_params(
            self._cfg,
            environment_implementation=self._environment_specs.implementation,
        )

        # Helper function to create a trial configuration
        def create_trial_params(trial_idx):
            agent_actor_params = cogment.ActorParameters(
                cog_settings,
                name="player",
                class_name=PLAYER_ACTOR_CLASS,
                implementation="actors.tutorial.tutorial_1.SimpleBCActor",
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                ),
            )

            teacher_actor_params = cogment.ActorParameters(
                cog_settings,
                name=WEB_ACTOR_NAME,
                class_name=TEACHER_ACTOR_CLASS,
                implementation=HUMAN_ACTOR_IMPL,
                config=AgentConfig(
                    run_id=run_session.run_id,
                    environment_specs=self._environment_specs.serialize(),
                ),
            )

            return cogment.TrialParameters(
                cog_settings,
                environment_name="env",
                environment_implementation=self._environment_specs.implementation,
                environment_config=EnvironmentConfig(
                    run_id=run_session.run_id, render=True, seed=self._cfg.seed + trial_idx
                ),
                actors=[agent_actor_params, teacher_actor_params],
            )

        # Rollout a bunch of trials
        for (step_idx, _trial_id, _trial_idx, sample,) in run_session.start_and_await_trials(
            trials_id_and_params=[
                (f"{run_session.run_id}_{trial_idx}", create_trial_params(trial_idx))
                for trial_idx in range(self._cfg.num_trials)
            ],
            sample_producer_impl=self.sample_producer,
            num_parallel_trials=1,
        ):
            log.info(f"[{step_idx}] - Got sample [{sample}]")
