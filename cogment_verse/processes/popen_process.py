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

from threading import Thread
import logging
import signal
import subprocess

from .cogment_verse_process import CogmentVerseProcess

log = logging.getLogger(__name__)

TERMINATION_TIMEOUT_SECONDS = 10


def main(name, args, cwd, env, on_ready, on_log, on_awaiting_ready):
    log.debug(f"Launching subprocess [{name}] with args {args}...")
    with subprocess.Popen(
        args=args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as external_process:

        def termination_handler(_signum, _frame):
            log.info(f"terminating [{name}]")
            external_process.terminate()
            external_process.send_signal(signal.SIGINT)
            try:
                external_process.wait(TERMINATION_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                log.warning(
                    f"Subprocess [{name}] didn't terminate cleany under {TERMINATION_TIMEOUT_SECONDS}s, killing it"
                )
                external_process.kill()

        signal.signal(signal.SIGINT, termination_handler)
        signal.signal(signal.SIGTERM, termination_handler)

        def consume_stdout():
            for log_line in external_process.stdout:
                on_log(name, log_line.decode().strip(" \n\r\t"))

        log_consumer = Thread(name=f"{name}_log_consumer", target=consume_stdout)
        log_consumer.start()

        on_awaiting_ready()
        on_ready()

        external_process.wait()


def default_on_awaiting_ready():
    pass


class PopenProcess(CogmentVerseProcess):
    def __init__(self, name, args, on_log, on_awaiting_ready=default_on_awaiting_ready, cwd=".", env=None):
        super().__init__(
            name=name, target=main, args=args, on_log=on_log, on_awaiting_ready=on_awaiting_ready, cwd=cwd, env=env
        )
