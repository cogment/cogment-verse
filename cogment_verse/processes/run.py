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
from omegaconf import OmegaConf

from ..run import RunSession
from ..services_directory import ServiceType
from ..utils.import_class import import_class
from .cogment_py_sdk_process import CogmentPySdkProcess

log = logging.getLogger(__name__)


def run_main(
    name,  # pylint: disable=unused-argument
    on_ready,
    run_cfg,
    services_directory,
    specs_filename,  # pylint: disable=unused-argument
    work_dir,
):
    async def run_main_async():
        # Importing 'specs' only in the subprocess (i.e. where generate has been properly executed)
        # pylint: disable-next=import-outside-toplevel
        from cogment_verse.specs import EnvironmentSpecs, cog_settings

        _run_cfg = run_cfg
        run_cls = import_class(_run_cfg.class_name)
        if hasattr(run_cls, "default_cfg"):
            _run_cfg = OmegaConf.merge(run_cls.default_cfg, _run_cfg)

        registered_environment_impl_names = services_directory.get_service_names(ServiceType.ENVIRONMENT)
        enviroment_impl_name = _run_cfg.get("environment", registered_environment_impl_names[0])

        try:
            environment_specs = EnvironmentSpecs.load(work_dir, enviroment_impl_name)
        except Exception as error:
            raise RuntimeError(f"Unable to start run, unknown environment: '{enviroment_impl_name}'") from error

        context = cogment.Context(cog_settings=cog_settings, user_id="cogment_verse_run")
        model_registry = await services_directory.get_model_registry(context)

        run_id = _run_cfg.run_id
        run = run_cls(cfg=_run_cfg, environment_specs=environment_specs)
        run_session = RunSession(
            run_cfg=_run_cfg, run_id=run_id, services_directory=services_directory, model_registry=model_registry
        )
        try:
            on_ready()
            log.info(f"Starting run [{run_id}] from [{_run_cfg.class_name}]")
            await run.impl(run_session)
        except Exception as error:
            run_session.terminate_failure()
            log.error(f"Error while executing run [{run_id}] from [{_run_cfg.class_name}]", exc_info=error)
            raise error

        run_session.terminate_success()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_main_async())
    except KeyboardInterrupt:
        log.warning("interrupted by user")
        sys.exit(-1)


def create_run_process(work_dir, specs_filename, services_directory, run_cfg):
    return CogmentPySdkProcess(
        name="run",
        work_dir=work_dir,
        specs_filename=specs_filename,
        main=run_main,
        services_directory=services_directory,
        run_cfg=run_cfg,
    )
