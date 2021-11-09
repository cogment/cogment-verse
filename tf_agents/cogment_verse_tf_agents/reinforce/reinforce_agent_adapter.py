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

from data_pb2 import RunConfig

from cogment_verse_tf_agents.reinforce.reinforce import ReinforceAgent
from cogment_verse_tf_agents.reinforce.sample_producer import sample_producer
from cogment_verse_tf_agents.reinforce.training_run import create_training_run
from cogment_verse_tf_agents.wrapper import format_legal_moves, cog_action_from_tf_action, tf_obs_from_cog_obs

from cogment_verse import AgentAdapter

import cogment

from prometheus_client import Summary
import torch

import logging

COMPUTE_NEXT_ACTION_TIME = Summary(
    "actor_implementation_compute_next_action_seconds",
    "Time spent computing the next action",
    ["impl_name"],
)

log = logging.getLogger(__name__)

# pylint: disable=W0212
# pylint: disable=W0622
# pylint: disable=C0103
class ReinforceAgentAdapter(AgentAdapter):
    def _create(self, model_id, **kwargs):
        return ReinforceAgent(id=model_id, **kwargs)

    @staticmethod
    def extract_agent_params(model):
        agent_params = {}
        agent_params["id"] = model.id
        agent_params["_params"] = model._params
        agent_params["_lr_schedule"] = model._lr_schedule
        agent_params["model_params"] = model._model.get_weights()
        return agent_params

    def _load(self, model_id, version_number, version_user_data, model_data_f):

        agent_params = torch.load(model_data_f)
        reinforce_agent = ReinforceAgent(id=agent_params["id"], obs_dim=agent_params["_params"]["obs_dim"],
                        act_dim=agent_params["_params"]["act_dim"],lr_schedule=agent_params["_lr_schedule"],
                        gamma=agent_params["_params"]["gamma"],max_replay_buffer_size=agent_params["_params"]["max_replay_buffer_size"],
                        seed=agent_params["_params"]["seed"], model_params=agent_params["model_params"])

        return reinforce_agent

    def _save(self, model, model_data_f):
        agent_params = self.extract_agent_params(model)
        torch.save(agent_params, model_data_f)
        return {}

    def _create_actor_implementations(self):
        async def impl(actor_session):
            log.debug(f"[reinforce - {actor_session.name}] trial {actor_session.get_trial_id()} starts")
            actor_session.start()

            # Retrieve the latest version of the agent model (asynchronous so needs to be done after the start)
            model, version_info = await self.retrieve_version(
                actor_session.config.model_id, actor_session.config.model_version
            )

            version_number = version_info["version_number"]
            log.debug(
                f"[reinforce - {actor_session.name}] model {actor_session.config.model_id}@v{version_number} retrieved"
            )

            actor_map = {actor.actor_name: idx for idx, actor in enumerate(actor_session.get_active_actors())}
            actor_index = actor_map[actor_session.name]

            total_reward = 0

            async for event in actor_session.event_loop():
                for reward in event.rewards:
                    total_reward += reward.value

                if event.observation and event.type == cogment.EventType.ACTIVE:
                    with COMPUTE_NEXT_ACTION_TIME.labels(impl_name="reinforce").time():
                        obs = event.observation.snapshot
                        obs = tf_obs_from_cog_obs(obs)

                        obs_input = obs["vectorized"]
                        legal_moves_input = format_legal_moves(
                            obs["legal_moves_as_int"], actor_session.config.num_action
                        )

                        if obs["current_player"] != actor_index:
                            # Use -1 to indicate no action, since not active player
                            action = -1
                        else:
                            action = model.act(obs_input, legal_moves_input)

                        cog_action = cog_action_from_tf_action(action)
                        actor_session.do_action(cog_action)

        return {"reinforce": (impl, ["agent"])}

    def _create_run_implementations(self):
        return {"reinforce_training": (sample_producer, create_training_run(self), RunConfig())}
