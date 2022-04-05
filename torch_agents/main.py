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
import json
import logging
import os
import sys
import torch.multiprocessing as mp
from dotenv import load_dotenv
import cogment

from cogment_verse import RunContext
from cogment_verse_torch_agents.hive_adapter.hive_agent_adapter import HiveAgentAdapter
from cogment_verse_torch_agents.muzero.adapter import MuZeroAgentAdapter
from cogment_verse_torch_agents.simple_a2c.simple_a2c_agent import SimpleA2CAgentAdapter
from cogment_verse_torch_agents.simple_bc import SimpleBCAgentAdapter
from cogment_verse_torch_agents.hf_sb3.sb3_adapter import SimpleSB3AgentAdapter
from cogment_verse_torch_agents.selfplay_td3.selfplay_agent import SelfPlayAgentAdapter

import cog_settings

load_dotenv()

PORT = int(os.getenv("COGMENT_VERSE_TORCH_AGENTS_PORT", "9000"))
PROMETHEUS_PORT = int(os.getenv("COGMENT_VERSE_TORCH_AGENTS_PROMETHEUS_PORT", "8000"))

TRIAL_DATASTORE_ENDPOINT = os.getenv("COGMENT_VERSE_TRIAL_DATASTORE_ENDPOINT")
MODEL_REGISTRY_ENDPOINT = os.getenv("COGMENT_VERSE_MODEL_REGISTRY_ENDPOINT")
ORCHESTRATOR_ENDPOINT = os.getenv("COGMENT_VERSE_ORCHESTRATOR_ENDPOINT")
ACTOR_ENDPOINTS = json.loads(os.getenv("COGMENT_VERSE_ACTOR_ENDPOINTS"))
ENVIRONMENT_ENDPOINTS = json.loads(os.getenv("COGMENT_VERSE_ENVIRONMENT_ENDPOINTS"))


logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


async def main():
    context = RunContext(
        cog_settings=cog_settings,
        user_id="cogment_verse_torch_agents",
        services_endpoints={
            "orchestrator": "grpc://" + ORCHESTRATOR_ENDPOINT,
            "trial_datastore": TRIAL_DATASTORE_ENDPOINT,
            "model_registry": MODEL_REGISTRY_ENDPOINT,
            **ACTOR_ENDPOINTS,
            **ENVIRONMENT_ENDPOINTS,
        },
    )

    hive_adapter = HiveAgentAdapter()
    hive_adapter.register_implementations(context)

    simple_a2c_adapter = SimpleA2CAgentAdapter()
    simple_a2c_adapter.register_implementations(context)

    muzero_adapter = MuZeroAgentAdapter()
    muzero_adapter.register_implementations(context)

    simple_bc_adapter = SimpleBCAgentAdapter()
    simple_bc_adapter.register_implementations(context)

    simple_sb3_adapter = SimpleSB3AgentAdapter()
    simple_sb3_adapter.register_implementations(context)

    selfplay_td3_adapter = SelfPlayAgentAdapter()
    selfplay_td3_adapter.register_implementations(context)

    log.info(f"Torch agents service starts on {PORT}...")

    await context.serve_all_registered(cogment.ServedEndpoint(port=PORT), prometheus_port=PROMETHEUS_PORT)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # suggested fix https://github.com/pytorch/pytorch/issues/67864
    mp.set_sharing_strategy("file_system")
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        log.error("process interrupted")
        sys.exit(-1)
