# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import json
import logging
import os
import re
import time

from ..constants import COGMENT_VERSION
from ..utils.download_cogment import download_cogment
from .popen_process import PopenProcess

log = logging.getLogger(__name__)

LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

RESERVED_MSG_KEYS = ["msg", "level", "component", "time"]


def on_cogment_log(name, log_line):
    try:
        parsed_msg = json.loads(log_line)
    except json.JSONDecodeError:
        log.warning(f"Unstructured log from [cogment.{name}] - {log_line}")
        return

    logger_name = "cogment"
    component = parsed_msg.get("component", None)
    if component is not None:
        logger_name = logger_name + "." + re.sub(r"[/\s]", ".", component)
    logger = logging.getLogger(logger_name)

    level_str = parsed_msg.get("level", "info")
    if level_str not in LOG_LEVELS:
        raise RuntimeError(f"Unknown log level [{level_str}].")
    level = LOG_LEVELS[level_str]

    if logger.isEnabledFor(level):
        msg_str = parsed_msg.get("msg", "").strip(" \n\r")
        for key, value in parsed_msg.items():
            if key not in RESERVED_MSG_KEYS:
                msg_str += f" [{key}={value}]"

        logger.log(level, msg_str)


def on_awaiting_cogment_ready():
    time.sleep(5)  # TODO replace with the status file (will require support for the trial datastore and model registry)


class CogmentCliProcess(PopenProcess):
    def __init__(self, name, work_dir, cli_args):
        cogment_path = download_cogment(output_dir=os.path.join(work_dir, "bin"), desired_version=COGMENT_VERSION)

        super().__init__(
            name=name,
            args=[cogment_path, *cli_args],
            on_log=on_cogment_log,
            on_awaiting_ready=on_awaiting_cogment_ready,
        )
