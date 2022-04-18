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
from cogment_verse_tf_agents.reinforce.reinforce import ReinforceAgent
from cogment_verse_tf_agents.reinforce.sample_producer import sample_producer
from cogment_verse_tf_agents.reinforce.training_run import create_training_run
from cogment_verse_tf_agents.wrapper import cog_action_from_tf_action, tf_obs_from_cog_obs
from data_pb2 import RunConfig
from prometheus_client import Summary

COMPUTE_NEXT_ACTION_TIME = Summary(
    "actor_implementation_compute_next_action_seconds",
    "Time spent computing the next action",
    ["impl_name"],
)

log = logging.getLogger(__name__)

# pylint: disable=W0212
# pylint: disable=W0622
# pylint: disable=C0103
# pylint: disable=arguments-differ
class ReinforceAgentAdapter(AgentAdapter):
    def _create(self, model_id, environment_specs, **kwargs):
        model = ReinforceAgent(
            id=model_id, obs_dim=environment_specs.num_input, act_dim=environment_specs.num_action, **kwargs
        )
        model_user_data = {
            "environment_implementation": environment_specs.implementation,
            "num_input": environment_specs.num_input,
            "num_action": environment_specs.num_action,
        }
        return model, model_user_data

    def _load(self, model_id, version_number, model_user_data, version_user_data, model_data_f, **kwargs):
        return ReinforceAgent.load(model_data_f, id=model_id, **version_user_data)

    def _save(self, model, model_user_data, model_data_f, **kwargs):
        return model.save(model_data_f)

    def _create_actor_implementations(self):
        async def impl(actor_session):
            log.debug(f"[reinforce - {actor_session.name}] trial {actor_session.get_trial_id()} starts")
            actor_session.start()

            # Retrieve the latest version of the agent model (asynchronous so needs to be done after the start)
            model, _, version_info = await self.retrieve_version(
                actor_session.config.model_id, actor_session.config.model_version
            )

            version_number = version_info["version_number"]
            log.debug(
                f"[reinforce - {actor_session.name}] model {actor_session.config.model_id}@v{version_number} retrieved"
            )

            actor_index = actor_session.config.actor_index

            total_reward = 0

            async for event in actor_session.all_events():
                for reward in event.rewards:
                    total_reward += reward.value

                if event.observation and event.type == cogment.EventType.ACTIVE:
                    with COMPUTE_NEXT_ACTION_TIME.labels(impl_name="reinforce").time():
                        obs = event.observation.snapshot
                        obs = tf_obs_from_cog_obs(obs)

                        obs_input = obs["vectorized"]

                        if obs["current_player"] != actor_index:
                            # Use -1 to indicate no action, since not active player
                            action = -1
                        else:
                            action = model.act(obs_input)

                        cog_action = cog_action_from_tf_action(action)
                        actor_session.do_action(cog_action)

        return {"reinforce": (impl, ["agent"])}

    def _create_run_implementations(self):
        return {"reinforce_training": (sample_producer, create_training_run(self), RunConfig())}
