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

import logging

import cogment
from cogment_verse import AgentAdapter
from cogment_verse_torch_agents.atari_cnn import NatureAtariDQNModel
from cogment_verse_torch_agents.hive_adapter.sample_producer import sample_producer
from cogment_verse_torch_agents.hive_adapter.training_run import create_training_run
from cogment_verse_torch_agents.third_party.hive.ddpg import DDPGAgent
from cogment_verse_torch_agents.third_party.hive.dqn import DQNAgent
from cogment_verse_torch_agents.third_party.hive.rainbow import RainbowDQNAgent
from cogment_verse_torch_agents.third_party.td3.td3 import TD3Agent
from cogment_verse_torch_agents.wrapper import cog_action_from_torch_action, format_legal_moves, torch_obs_from_cog_obs
from data_pb2 import RunConfig
from prometheus_client import Summary

COMPUTE_NEXT_ACTION_TIME = Summary(
    "actor_implementation_compute_next_action_seconds",
    "Time spent computing the next action",
    ["impl_name"],
)

log = logging.getLogger(__name__)


class HiveAgentAdapter(AgentAdapter):
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
    def _create(self, model_id, impl_name, environment_specs, **kwargs):
        model = self.agent_class_from_impl_name(impl_name)(
            id=model_id, obs_dim=environment_specs.num_input, act_dim=environment_specs.num_action, **kwargs
        )

        model_user_data = {
            "environment_implementation": environment_specs.implementation,
            "num_input": environment_specs.num_input,
            "num_action": environment_specs.num_action,
        }

        return model, model_user_data

    def _load(self, model_id, version_number, model_user_data, version_user_data, model_data_f, **kwargs):
        model = self.agent_class_from_impl_name(model_user_data["agent_implementation"])(
            id=model_id,
            obs_dim=int(model_user_data["num_input"]),
            act_dim=int(model_user_data["num_action"]),
        )

        model.load(model_data_f)
        model.set_version_info(version_number, None)

        return model

    def _save(self, model, model_user_data, model_data_f, **kwargs):
        model.save(model_data_f)

        # pylint: disable=protected-access
        return {}

    def _create_actor_implementations(self):
        def create_actor_impl(impl_name):
            async def impl(actor_session):
                log.debug(f"[{impl_name} - {actor_session.name}] trial {actor_session.get_trial_id()} starts")
                actor_session.start()

                # Retrieve the latest version of the agent model (asynchronous so needs to be done after the start)
                model, _, version_info = await self.retrieve_version(
                    actor_session.config.model_id, actor_session.config.model_version
                )

                version_number = version_info["version_number"]
                log.debug(
                    f"[{impl_name} - {actor_session.name}] model {actor_session.config.model_id}@v{version_number} retrieved"
                )

                actor_index = actor_session.config.actor_index

                total_reward = 0

                async for event in actor_session.all_events():
                    for reward in event.rewards:
                        total_reward += reward.value

                    if event.observation and event.type == cogment.EventType.ACTIVE:
                        with COMPUTE_NEXT_ACTION_TIME.labels(impl_name=impl_name).time():
                            obs = event.observation.snapshot
                            obs = torch_obs_from_cog_obs(obs)

                            obs_input = obs["vectorized"]
                            legal_moves_input = format_legal_moves(
                                obs["legal_moves_as_int"], actor_session.config.environment_specs.num_action
                            )

                            if obs["current_player"] != actor_index:
                                # Use -1 to indicate no action, since not active player
                                action = -1
                            else:
                                action = model.act(obs_input, legal_moves_input)

                            cog_action = cog_action_from_torch_action(action)
                            actor_session.do_action(cog_action)

            return impl

        return {impl_name: (create_actor_impl(impl_name), ["agent"]) for impl_name in self._agent_classes}

    def _create_run_implementations(self):
        return {"cogment_verse_run_impl": (sample_producer, create_training_run(self), RunConfig())}
