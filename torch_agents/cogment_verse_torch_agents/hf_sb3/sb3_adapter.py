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

import asyncio
import logging

import cogment

import torch
from cogment_verse import AgentAdapter

from cogment_verse_torch_agents.utils.tensors import tensor_from_cog_obs
from data_pb2 import AgentAction

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

log = logging.getLogger(__name__)

# pylint: disable=arguments-differ
class SimpleSB3AgentAdapter(AgentAdapter):
    def __init__(self):
        super().__init__()
        self._dtype = torch.float

    @staticmethod
    async def run_async(func, *args):
        """Run a given function asynchronously in the default thread pool"""
        event_loop = asyncio.get_running_loop()
        return await event_loop.run_in_executor(None, func, *args)

    def _create_actor_implementations(self):
        async def impl(actor_session):
            actor_session.start()

            checkpoint = load_from_hub(
                repo_id=actor_session.config.hf_hub_model.repo_id,
                filename=actor_session.config.hf_hub_model.filename,
            )

            model = PPO.load(checkpoint)

            @torch.no_grad()
            def compute_action(event):
                obs = tensor_from_cog_obs(event.observation.snapshot, dtype=self._dtype)
                obs = torch.unsqueeze(obs, dim=0)
                action = model.predict(obs)

                return action

            async for event in actor_session.event_loop():
                if event.observation and event.type == cogment.EventType.ACTIVE:
                    action = await self.run_async(compute_action, event)
                    actor_session.do_action(AgentAction(discrete_action=action[0]))

        return {
            "simple_sb3": (impl, ["agent"]),
        }

    def _create_run_implementations(self):
        return {}
