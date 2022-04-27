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
import numbers
import os

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from prometheus_client import Counter, Summary

from mlflow.entities import Metric, Param, RunStatus
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

log = logging.getLogger(__name__)

EXPERIMENT_TRACKER_LOG_METRICS_TIME = Summary(
    "experiment_tracker_log_metrics_seconds", "Time spent logging training metrics"
)
EXPERIMENT_TRACKER_METRICS_LOGGED_COUNTER = Counter(
    "experiment_tracker_metrics_logged", "Counter of individual logged metrics"
)

MAX_METRICS_BATCH_SIZE = 1000  # MLFlow only accepts at most 1000 metrics per batch

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


def make_dict(ignore_non_numbers, *args, **kwargs):
    res = dict(kwargs)
    for arg in args:
        if isinstance(arg, Message):
            arg = MessageToDict(arg, preserving_proto_field_name=True, including_default_value_fields=False)
        if isinstance(arg, dict):
            for key, value in arg.items():
                if ignore_non_numbers and not isinstance(value, numbers.Number):
                    break
                if key in res:
                    raise RuntimeError(
                        f"Trying to set duplicate key [{key}] to [{value}], while already set to [{res[key]}]"
                    )
                res[key] = value
        else:
            raise RuntimeError(f"Unsupported argument of type [{type(arg)}]")
    return res


class MlflowExperimentTracker:
    def __init__(self, experiment_id, run_id, flush_frequency=5):
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._mlflow_exp_id = None
        self._mlflow_run_id = None
        self._metrics_buffer = []
        self._flush_metrics_worker_frequency = flush_frequency
        self._flush_metrics_worker = None

    def _get_mlflow_client(self):
        # This is automagically configured by the environment variable MLFLOW_TRACKING_URI
        client = MlflowClient()
        if not self._mlflow_exp_id:
            mlflow_experiment_name = f"/{self._experiment_id}"
            experiment = client.get_experiment_by_name(mlflow_experiment_name)
            if experiment is not None:
                self._mlflow_exp_id = experiment.experiment_id
            else:
                log.info(f"Experiment with name '{mlflow_experiment_name}' not found. Creating it.")
                self._mlflow_exp_id = client.create_experiment(mlflow_experiment_name)

        if not self._mlflow_run_id:
            run = client.create_run(self._mlflow_exp_id, tags={MLFLOW_RUN_NAME: self._run_id})
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
                except Exception as err:
                    log.warning(
                        f"Error while sending metrics to mlflow server {MLFLOW_TRACKING_URI}. Will retry later in {self._flush_metrics_worker_frequency}s.",
                        err,
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

    def log_metrics(self, step_timestamp, step_idx, *args, **kwargs):
        EXPERIMENT_TRACKER_METRICS_LOGGED_COUNTER.inc(len(kwargs))
        self._metrics_buffer.extend(
            [Metric(key, value, step_timestamp, step_idx) for key, value in make_dict(True, *args, **kwargs).items()]
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
