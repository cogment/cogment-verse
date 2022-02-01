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
import os
import sys

import cog_settings
import cogment
from cogment_verse_environment.environment_adapter import EnvironmentAdapter
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("COGMENT_VERSE_ENVIRONMENT_PORT", "9000"))
PROMETHEUS_PORT = int(os.getenv("COGMENT_VERSE_ENVIRONMENT_PROMETHEUS_PORT", "8000"))

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


async def main():
    context = cogment.Context(cog_settings=cog_settings, user_id="cogment_verse_environment")

    environment_adapter = EnvironmentAdapter()
    environment_adapter.register_implementations(context)

    log.info(f"Environment service starting on port {PORT}...")
    await context.serve_all_registered(cogment.ServedEndpoint(port=PORT), prometheus_port=PROMETHEUS_PORT)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("process interrupted")
        sys.exit(-1)
