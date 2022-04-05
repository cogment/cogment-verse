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
import datetime
import functools
import json
import os

import click
import yaml
from google.protobuf.json_format import ParseDict
from run_controller import RunController, RunStatus
# from google.protobuf.json_format import MessageToDict

# pylint: disable=too-many-arguments,import-outside-toplevel

RUN_ENDPOINTS = json.loads(os.getenv("COGMENT_VERSE_RUN_ENDPOINTS", "{}"))


def import_class(class_name):
    from importlib import import_module

    module_path, class_name = class_name.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def load_run_params(params_path, params_name):
    with open(params_path, encoding="utf-8") as f:
        run_params = yaml.safe_load(f)

    if params_name not in run_params:
        raise Exception(f"Undefined run '{params_name}'")

    run_implementation = run_params[params_name]["implementation"]
    run_config_kwargs = run_params[params_name]["config"]
    run_config_class_name = run_config_kwargs["class_name"]
    del run_config_kwargs["class_name"]
    run_config_class = import_class(run_config_class_name)
    run_config = ParseDict(run_config_kwargs, run_config_class())
    return run_implementation, run_config


def make_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def get_controller(impl_name=None):
    if not impl_name:
        return RunController(
            endpoints={endpoint for impl_endpoints in RUN_ENDPOINTS.values() for endpoint in impl_endpoints}
        )
    return RunController(endpoints=RUN_ENDPOINTS[impl_name])


def pretty_print(val):
    def converter(val):
        if isinstance(val, datetime.datetime):
            return val.__str__()
        if isinstance(val, RunStatus):
            return val.name
        return None

    click.echo(json.dumps(val, indent=2, default=converter))


@click.group()
def cli():
    pass


@cli.command(name="list", help="List runs")
@make_sync
async def retrieve_list():
    ongoing_runs = await get_controller().list_runs()
    pretty_print(ongoing_runs)


@cli.command(help="Start a run with the given params")
@click.argument("params")
@click.option("--run_id", default=None, help="Unique identifier for the run")
@click.option(
    "--params_path",
    default="./run_params.yaml",
    help="Path for the parameters definitions yaml file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
)
@make_sync
async def start(run_id, params_path, params):
    run_implementation, run_config = load_run_params(params_path, params)
    # click.echo(f"- id: {run_id}")
    # click.echo(f"- params_name: {params}")
    # click.echo(f"- implementation: {run_implementation}")
    # click.echo("- config:")
    # pretty_print(MessageToDict(run_config))
    run = await get_controller(run_implementation).start_run(
        name=params, implementation=run_implementation, config=run_config, run_id=run_id
    )
    pretty_print(run)


@cli.command(help="Terminate a run")
@click.argument("run_id")
@make_sync
async def terminate(run_id):
    run = await get_controller().terminate_run(run_id)
    pretty_print(run)


if __name__ == "__main__":
    cli()
