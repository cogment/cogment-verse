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

import cogment

from .cogment_py_sdk_process import CogmentPySdkProcess
from ..utils.import_class import import_class
from ..utils.get_implementation_name import get_implementation_name
from ..services_directory import ServiceType

log = logging.getLogger(__name__)


def actor_main(
    actor_cfg,
    model_registry,
    name,  # pylint: disable=unused-argument
    on_ready,
    specs_filename,  # pylint: disable=unused-argument
    work_dir,  # pylint: disable=unused-argument
):
    # Importing 'specs' only in the subprocess (i.e. where generate has been properly executed)
    # pylint: disable-next=import-outside-toplevel
    from cogment_verse.specs import cog_settings

    async def actor_main_async():
        actor_cls = import_class(actor_cfg.class_name)
        actor = actor_cls(actor_cfg)

        actor_implementation_name = get_implementation_name(actor)

        async def impl_wrapper(actor_session):
            actor_session.model_registry = model_registry
            await actor.impl(actor_session)

        context = cogment.Context(cog_settings=cog_settings, user_id="cogment_verse_actor")

        context.register_actor(
            impl_name=actor_implementation_name, actor_classes=actor.get_actor_classes(), impl=impl_wrapper
        )

        log.info(f"Service actor [{actor_implementation_name}] starting on port [{actor_cfg.port}]...")

        on_ready()

        await context.serve_all_registered(
            cogment.ServedEndpoint(port=actor_cfg.port), prometheus_port=actor_cfg.get("prometheus_port", 0)
        )

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(actor_main_async())
    except KeyboardInterrupt:
        log.warning("interrupted by user")
        sys.exit(-1)


def create_actor_service(work_dir, specs_filename, model_registry, actor_cfg, services_directory):
    process = CogmentPySdkProcess(
        name="actor",
        work_dir=work_dir,
        specs_filename=specs_filename,
        main=actor_main,
        model_registry=model_registry,
        actor_cfg=actor_cfg,
    )

    actor_cls = import_class(actor_cfg.class_name)
    actor = actor_cls(actor_cfg)

    # Register the actor implementation
    services_directory.add(
        service_type=ServiceType.ACTOR,
        service_name=get_implementation_name(actor),
        service_endpoint=f"grpc://localhost:{actor_cfg.port}",
    )

    return process
