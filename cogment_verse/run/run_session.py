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

import logging
import os
import time
import asyncio
from multiprocessing import Queue

from cogment_verse.model_registry import ModelRegistry

from ..mlflow_experiment_tracker import MlflowExperimentTracker
from .sample_producer_worker import start_sample_producer_worker
from .trial_runner_worker import start_trial_runner_worker

LOGLEVEL = os.environ.get("COGVERSE_LOG_LEVEL", "INFO").upper()
log = logging.getLogger(__name__)
log.setLevel(LOGLEVEL)


class RunSession:
    def __init__(self, run_cfg, run_id, services_directory, model_registry: ModelRegistry):
        self.run_id = run_id

        self._services_directory = services_directory
        self._step_idx = 0

        self._xp_tracker = MlflowExperimentTracker(
            experiment_id=run_cfg.class_name, run_id=run_id, mlflow_tracking_uri=run_cfg.mlflow_tracking_uri
        )

        self.model_registry = model_registry

    async def start_and_await_trials(self, trials_id_and_params, sample_producer_impl, num_parallel_trials=10):
        trial_started_queue = Queue()
        sample_queue = Queue()

        trial_runner_worker = start_trial_runner_worker(
            trials_id_and_params=trials_id_and_params,
            services_directory=self._services_directory,
            trial_started_queue=trial_started_queue,
            trial_ended_queue=None,
            num_parallel_trials=num_parallel_trials,
        )

        sample_producer_worker = start_sample_producer_worker(
            trial_started_queue=trial_started_queue,
            services_directory=self._services_directory,
            sample_queue=sample_queue,
            impl=sample_producer_impl,
        )

        try:
            log.debug(f"Starting sample reading")
            while True:
                sample_queue_event = await asyncio.get_running_loop().run_in_executor(None, sample_queue.get)
                if sample_queue_event.done:
                    break
                self._step_idx += 1
                yield (
                    self._step_idx,
                    sample_queue_event.trial_id,
                    sample_queue_event.trial_idx,
                    sample_queue_event.sample,
                )

            log.debug(f"Sample reading is done at step [{self._step_idx}]")
            await asyncio.get_running_loop().run_in_executor(None, trial_runner_worker.join)
            await asyncio.get_running_loop().run_in_executor(None, sample_producer_worker.join)
            log.debug(f"All processes joined")

        except RuntimeError as error:
            trial_runner_worker.terminate()
            sample_producer_worker.terminate()
            trial_runner_worker.join()
            sample_producer_worker.join()
            raise error

    def log_params(self, *args, **kwargs):
        self._xp_tracker.log_params(*args, **kwargs)

    def log_metrics(self, **kwargs):
        self._xp_tracker.log_metrics(step_timestamp=int(time.time() * 1000), step_idx=self._step_idx, **kwargs)

    def terminate_failure(self):
        self._xp_tracker.terminate_failure()

    def terminate_success(self):
        self._xp_tracker.terminate_success()
