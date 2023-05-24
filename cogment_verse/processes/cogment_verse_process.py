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

from multiprocessing import Process, SimpleQueue

from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log


def wrapped_target(target, hydra_cfg_job_logging, hydra_cfg_verbose, *args, **kwargs):
    configure_log(hydra_cfg_job_logging, hydra_cfg_verbose)
    return target(*args, **kwargs)


class SimpleSignal:
    def __init__(self):
        self._q = SimpleQueue()

    def trigger(self):
        self._q.put(True)

    def await_trigger(self):
        self._q.get()


class CogmentVerseProcess(Process):
    def __init__(self, name, target, **kwargs):
        self._ready_signal = SimpleSignal()
        hydra_cfg = HydraConfig.get()
        super().__init__(
            name=name,
            target=wrapped_target,
            kwargs={
                **kwargs,
                "target": target,
                "hydra_cfg_job_logging": hydra_cfg.job_logging,
                "hydra_cfg_verbose": hydra_cfg.verbose,
                "name": name,
                "on_ready": self._ready_signal.trigger,
            },
        )

    def await_ready(self):
        self._ready_signal.await_trigger()
