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

import asyncio
import logging
import os

from mlflow.entities import Metric, Param, RunStatus
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Summary

from cogment_verse.experiment_tracker.simple_experiment_tracker import make_dict
from cogment_verse.utils.errors import CogmentVerseError

log = logging.getLogger(__name__)

EXPERIMENT_TRACKER_LOG_METRICS_TIME = Summary(
    "experiment_tracker_log_metrics_seconds", "Time spent logging training metrics"
)
EXPERIMENT_TRACKER_METRICS_LOGGED_COUNTER = Counter(
    "experiment_tracker_metrics_logged", "Counter of individual logged metrics"
)

MAX_METRICS_BATCH_SIZE = 1000  # MLFlow only accepts at most 1000 metrics per batch


class MlflowExperimentTracker:
    def __init__(self, exp_tracker_cfg, experiment_id, run_id):
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._mlflow_tracking_uri = exp_tracker_cfg.mlflow_tracking_uri
        self._mlflow_exp_id = None
        self._mlflow_run_id = None
        self._metrics_buffer = []
        self._flush_metrics_worker_frequency = exp_tracker_cfg.flush_frequency
        self._flush_metrics_worker = None

        os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] = str(exp_tracker_cfg.request_backoff_factor)
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(exp_tracker_cfg.request_max_retries)
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(exp_tracker_cfg.request_timeout)

    def __del__(self):
        self._stop_flush_metrics_worker()

    def _get_mlflow_client(self):

        client = MlflowClient(tracking_uri=self._mlflow_tracking_uri)

        if not self._mlflow_exp_id:
            mlflow_experiment_name = f"/{self._experiment_id}"
            try:
                experiment = client.get_experiment_by_name(mlflow_experiment_name)
            except MlflowException:
                raise CogmentVerseError(
                    "Mlflow server is not responding. Make sure it is launched in a separate terminal."
                ) from None

            if experiment is not None:
                self._mlflow_exp_id = experiment.experiment_id
            else:
                log.info(f"Experiment with name '{mlflow_experiment_name}' not found. Creating it.")
                self._mlflow_exp_id = client.create_experiment(mlflow_experiment_name)

        if not self._mlflow_run_id:
            run = client.create_run(experiment_id=self._mlflow_exp_id, run_name=self._run_id)
            self._mlflow_run_id = run.info.run_id

        return client

    def _flush_metrics(self):
        client = self._get_mlflow_client()
        while len(self._metrics_buffer) > 0:
            metrics_batch = self._metrics_buffer[:MAX_METRICS_BATCH_SIZE]
            with EXPERIMENT_TRACKER_LOG_METRICS_TIME.time():
                client.log_batch(
                    run_id=self._mlflow_run_id,
                    metrics=metrics_batch,
                )
            self._metrics_buffer = self._metrics_buffer[MAX_METRICS_BATCH_SIZE:]

    def _start_flush_metrics_worker(self):
        if self._flush_metrics_worker is not None:
            return

        async def worker():
            while True:
                try:
                    self._flush_metrics()
                except asyncio.CancelledError as cancelled_error:
                    # Raising cancellation
                    raise cancelled_error
                except Exception as error:  # pylint: disable=broad-except
                    log.warning(
                        f"Error while sending metrics to mlflow server [{self._mlflow_tracking_uri}]. Will retry later in {self._flush_metrics_worker_frequency}s.",
                        error,
                    )
                await asyncio.sleep(self._flush_metrics_worker_frequency)

        self._flush_metrics_worker = asyncio.create_task(worker())

    def _stop_flush_metrics_worker(self):
        if self._flush_metrics_worker is not None:
            self._flush_metrics_worker.cancel()
            # We don't really need to await for the termination here
            self._flush_metrics_worker = None

    def log_params(self, *args, **kwargs):
        self._get_mlflow_client().log_batch(
            run_id=self._mlflow_run_id,
            params=[Param(key, str(value)) for key, value in make_dict(False, *args, **kwargs).items()],
        )

    def log_metrics(self, step_timestamp, step_idx, **kwargs):
        EXPERIMENT_TRACKER_METRICS_LOGGED_COUNTER.inc(len(kwargs))
        self._metrics_buffer.extend(
            [Metric(key, value, step_timestamp, step_idx) for key, value in make_dict(True, **kwargs).items()]
        )
        self._start_flush_metrics_worker()

    def terminate_failure(self):
        self._stop_flush_metrics_worker()
        self._flush_metrics()
        self._get_mlflow_client().set_terminated(
            run_id=self._mlflow_run_id, status=RunStatus.to_string(RunStatus.FAILED)
        )

    def terminate_success(self):
        self._stop_flush_metrics_worker()
        self._flush_metrics()
        self._get_mlflow_client().set_terminated(
            run_id=self._mlflow_run_id, status=RunStatus.to_string(RunStatus.FINISHED)
        )
