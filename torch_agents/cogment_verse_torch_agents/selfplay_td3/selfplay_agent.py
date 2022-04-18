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
from data_pb2 import SelfPlayTD3TrainingRunConfig
from cogment_verse_torch_agents.selfplay_td3.selfplay_td3 import SelfPlayTD3
from cogment_verse_torch_agents.selfplay_td3.selfplay_sample_producer import sample_producer
from cogment_verse_torch_agents.selfplay_td3.selfplay_training_run import create_training_run
from cogment_verse_torch_agents.selfplay_td3.wrapper import (
    cog_action_from_tensor,
    tensor_from_cog_state,
    tensor_from_cog_goal,
    tensor_from_cog_grid,
)
from cogment_verse import AgentAdapter
import cogment
import torch

log = logging.getLogger(__name__)


# pylint: disable=W0212
# pylint: disable=W0221
# pylint: disable=W0622
# pylint: disable=C0103
class SelfPlayAgentAdapter(AgentAdapter):
    def _create(self, model_id, **kwargs):
        model = SelfPlayTD3(id=model_id, **kwargs)
        return model, kwargs

    def _load(self, model_id, version_number, model_user_data, version_user_data, model_data_f):
        return SelfPlayTD3.load(model_data_f, id=model_id, **model_user_data)

    def _save(self, model, model_user_data, model_data_f, **kwargs):
        return model.save(model_data_f)

    def _create_actor_implementations(self):
        async def impl(actor_session):
            log.debug(f"[selfplay_td3 - {actor_session.name}] trial {actor_session.get_trial_id()} starts")
            actor_session.start()

            # Retrieve the latest version of the agent model (asynchronous so needs to be done after the start)
            model, _, _ = await self.retrieve_version(
                actor_session.config.model_id,
                actor_session.config.model_version,
                environment_specs=actor_session.config.environment_specs,
            )

            agent = actor_session.config.model_id.split("_")[-1]
            total_reward = 0

            async for event in actor_session.all_events():
                for reward in event.rewards:
                    total_reward += reward.value

                if event.observation and event.type == cogment.EventType.ACTIVE:
                    obs = event.observation.snapshot
                    # process observation
                    # agent acts when its turn
                    if (obs.current_player == 1 and agent == "alice") or (obs.current_player == 0 and agent == "bob"):
                        state = tensor_from_cog_state(obs)
                        goal = tensor_from_cog_goal(obs)
                        grid = tensor_from_cog_grid(obs)
                        action = model.act(state, goal, grid)
                        cog_action = cog_action_from_tensor(action)
                        actor_session.do_action(cog_action)
                    else:  # agent stays put when not its turn
                        cog_action = cog_action_from_tensor(torch.tensor([0.0, 0.0]))
                        actor_session.do_action(cog_action)

        return {"selfplay_td3": (impl, ["agent"])}

    def _create_run_implementations(self):
        return {"selfplay_td3_training": (sample_producer, create_training_run(self), SelfPlayTD3TrainingRunConfig())}
