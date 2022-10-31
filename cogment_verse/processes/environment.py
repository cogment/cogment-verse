# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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
import sys
from hydra.core.hydra_config import HydraConfig  # pylint: disable=abstract-class-instantiated

import cogment

from .cogment_py_sdk_process import CogmentPySdkProcess
from ..utils.import_class import import_class
from ..utils.get_implementation_name import get_implementation_name
from ..services_directory import ServiceType

log = logging.getLogger(__name__)


def environment_main(
    environment_cfg,
    name,  # pylint: disable=unused-argument
    on_ready,
    specs_filename,  # pylint: disable=unused-argument
    work_dir,
):
    # Importing 'specs' only in the subprocess (i.e. where generate has been properly executed)
    # pylint: disable-next=import-outside-toplevel
    from cogment_verse.specs import cog_settings, save_environment_specs

    environment_cls = import_class(environment_cfg.class_name)
    env = environment_cls(environment_cfg)

    env_implementation_name = get_implementation_name(env)

    # Generate the environment specs if needed.
    save_environment_specs(work_dir, env_implementation_name, env.get_environment_specs())

    async def environment_main_async():
        context = cogment.Context(cog_settings=cog_settings, user_id="cogment_verse_environment")

        async def wrapped_impl(environment_session):
            try:
                await env.impl(environment_session)
            except KeyboardInterrupt:
                # Ignore this one, it's logged at a bunch of different places
                pass
            except Exception as error:
                log.error(
                    f"Uncaught error occured in implementation code for environment [{env_implementation_name}]",
                    exc_info=error,
                )
                raise

        context.register_environment(impl_name=env_implementation_name, impl=wrapped_impl)

        log.info(f"Environment [{env_implementation_name}] starting on port [{environment_cfg.port}]...")

        on_ready()

        await context.serve_all_registered(
            cogment.ServedEndpoint(port=environment_cfg.port), prometheus_port=environment_cfg.get("prometheus_port", 0)
        )

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(environment_main_async())
    except KeyboardInterrupt:
        log.warning("interrupted by user")
        sys.exit(-1)


def create_environment_service(work_dir, specs_filename, environment_cfg, services_directory):
    process = CogmentPySdkProcess(
        name="environment",
        work_dir=work_dir,
        specs_filename=specs_filename,
        main=environment_main,
        environment_cfg=environment_cfg,
    )

    environment_cls = import_class(environment_cfg.class_name)
    env = environment_cls(environment_cfg)

    # Register the environment
    services_directory.add(
        service_type=ServiceType.ENVIRONMENT,
        service_name=get_implementation_name(env),
        service_endpoint=f"grpc://localhost:{environment_cfg.port}",
    )

    return process
