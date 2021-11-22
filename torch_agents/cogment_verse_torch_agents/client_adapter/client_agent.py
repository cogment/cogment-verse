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

from data_pb2 import (
    AgentAction,
    TrialConfig,
    TrialActor,
    EnvConfig,
    ActorConfig,
    PipeGameRunConfig,
    RunConfig
)

from cogment_verse_torch_agents.atari_cnn import NatureAtariDQNModel
from cogment_verse_torch_agents.third_party.hive.ddpg import DDPGAgent
from cogment_verse_torch_agents.third_party.hive.dqn import DQNAgent
from cogment_verse_torch_agents.third_party.hive.rainbow import RainbowDQNAgent
from cogment_verse_torch_agents.third_party.td3.td3 import TD3Agent
from cogment_verse_torch_agents.wrapper import format_legal_moves, cog_action_from_torch_action, torch_obs_from_cog_obs
from cogment_verse_torch_agents.hive_adapter.sample_producer import sample_producer

from cogment_verse import AgentAdapter
from cogment_verse import MlflowExperimentTracker

from cogment.api.common_pb2 import TrialState
import cogment

import logging
import torch
import numpy as np

from collections import namedtuple

log = logging.getLogger(__name__)


# pylint: disable=arguments-differ
class ClientAgent(AgentAdapter):
    def __init__(self):
        super().__init__()
        self._agent_classes = {
            "td3": TD3Agent,
            "ddpg": DDPGAgent,
            "rainbowtorch": RainbowDQNAgent,
            "dqn": DQNAgent,
            "atari_cnn": NatureAtariDQNModel,
        }


    def agent_class_from_impl_name(self, impl_name):
        if impl_name not in self._agent_classes:
            raise ValueError(f"unknown hive actor implementation {impl_name}")
        return self._agent_classes[impl_name]

    def impl_name_from_agent_class(self, agent_class):
        impl_names = [
            impl_name for impl_name, self_agent_class in self._agent_classes.items() if self_agent_class == agent_class
        ]
        if len(impl_names) == 0:
            raise ValueError(f"unknown hive agent class {agent_class.__class__}")
        return impl_names[0]

    # pylint: disable=arguments-differ
    def _create(self, model_id, impl_name, **kwargs):
        return self.agent_class_from_impl_name(impl_name)(id=model_id, **kwargs)

    def _load(self, model_id, version_number, version_user_data, model_data_f):
        impl_name = version_user_data["impl_name"]
        model = self.agent_class_from_impl_name(impl_name)(
            id=model_id,
            obs_dim=int(version_user_data["obs_dim"]),
            act_dim=int(version_user_data["act_dim"]),
        )

        model.load(model_data_f)
        model.set_version_info(version_number, None)

        return model

    def _save(self, model, model_data_f):
        model.save(model_data_f)

        # pylint: disable=protected-access
        return {
            "impl_name": self.impl_name_from_agent_class(model.__class__),
            "obs_dim": model._params["obs_dim"],
            "act_dim": model._params["act_dim"],
        }

    def _create_actor_implementations(self):
        def create_actor_impl(impl_name):
            async def impl(actor_session):
                log.debug(f"[{impl_name} - {actor_session.name}] trial {actor_session.get_trial_id()} starts")
                actor_session.start()

                # Retrieve the latest version of the agent model (asynchronous so needs to be done after the start)
                model, version_info = await self.retrieve_version(
                    actor_session.config.model_id, actor_session.config.model_version
                )

                version_number = version_info["version_number"]
                log.debug(
                    f"[{impl_name} - {actor_session.name}] model {actor_session.config.model_id}@v{version_number} retrieved"
                )

                actor_map = {actor.actor_name: idx for idx, actor in enumerate(actor_session.get_active_actors())}
                actor_index = actor_map[actor_session.name]

                total_reward = 0

                async for event in actor_session.event_loop():
                    for reward in event.rewards:
                        total_reward += reward.value

                    if event.observation and event.type == cogment.EventType.ACTIVE:
                    
                        obs = event.observation.snapshot
                        obs = torch_obs_from_cog_obs(obs)
                        obs_input = obs["vectorized"]
                        
                        legal_moves_input = format_legal_moves(
                            obs["legal_moves_as_int"], actor_session.config.num_action
                        )

                        if obs["current_player"] != actor_index:
                            # Use -1 to indicate no action, since not active player
                            action = -1
                        else:
                            action = model.act(obs_input, legal_moves_input)

                        cog_action = cog_action_from_torch_action(action)
                        actor_session.do_action(cog_action)

            return impl

        print({impl_name: (create_actor_impl(impl_name), ["agent"]) for impl_name in self._agent_classes})
        return {impl_name: (create_actor_impl(impl_name), ["agent"]) for impl_name in self._agent_classes}

    def _create_run_implementations(self):
        
        async def run_impl(run_session):
            xp_tracker = MlflowExperimentTracker(run_session.params_name, run_session.run_id)

            # Initializing a model
            model_id = f"{run_session.run_id}_model"

            config = run_session.config
            model_id = config.model_id
            model_version = config.model_version

            # To investigate 
            trial_count = 1

            total_samples = 0
            for epoch in range(trial_count):
                # Rollout a bunch of trials
                observation = []
                action = []
                reward = []
                done = []
                epoch_last_step_idx = None
                epoch_last_step_timestamp = None
                async for (
                    step_idx,
                    step_timestamp,
                    _trial_id,
                    _tick_id,
                    sample,
                ) in run_session.start_trials_and_wait_for_termination(
                    trial_configs=[
                        TrialConfig(
                            run_id=run_session.run_id,
                            environment_config=config.environment,
                            actors=[
                                TrialActor(
                                    name="agent_1",
                                    actor_class="agent",
                                    implementation="dqn", ## Here to fix that 
                                    config=ActorConfig(
                                        model_id=model_id,
                                        model_version=model_version,
                                        num_input=config.actor.num_input,
                                        num_action=config.actor.num_action,
                                        env_type=config.environment.env_type,
                                        env_name=config.environment.env_name,
                                    ),
                                    
                                ),
                                TrialActor(
                                    name="web_actor",
                                    actor_class="pipe_player",
                                    implementation="client",
                                    config=ActorConfig(
                                        num_input=config.actor.num_input,
                                        num_action=config.actor.num_action,
                                        env_type=config.environment.env_type,
                                        env_name=config.environment.env_name,
                                    ),
                                )
                            ],
                        )
                        for trial_ids in range(trial_count)
                    ],
                    max_parallel_trials=trial_count,
                ):
                    (trial_observation, trial_action, trial_reward, trial_done) = sample
                    observation.extend(trial_observation)
                    action.extend(trial_action)
                    reward.extend(trial_reward)
                    done.extend(trial_done)
                    epoch_last_step_idx = step_idx
                    epoch_last_step_timestamp = step_timestamp

                    xp_tracker.log_metrics(step_timestamp, step_idx, total_reward=sum([r.item() for r in trial_reward]))

                total_samples += len(observation)

                # Convert the accumulated observation/action/reward over the epoch to tensors
                observation = torch.vstack(observation)
                action = torch.vstack(action)
                reward = torch.vstack(reward)
                done = torch.vstack(done)

               

                # Publish the newly trained version
                
                log.info(
                    f"[{run_session.params_name}/{run_session.run_id}] epoch #{epoch} finished ({total_samples} samples seen)"
                )

        log.info("******************************************")
        print("########### register human_dqn")
        return {
            "human_dqn": (
                sample_producer,
                run_impl,
                PipeGameRunConfig(environment=EnvConfig(
                        seed=12, env_type="pipe_world", env_name="PipeWorld", player_count=2, framestack=1
                    )),
                )
        }
