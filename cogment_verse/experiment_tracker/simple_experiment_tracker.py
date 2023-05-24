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
import numbers

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def make_dict(ignore_non_numbers, *args, **kwargs):
    res = dict(kwargs)
    for arg in args:
        if isinstance(arg, Message):
            arg = MessageToDict(arg, preserving_proto_field_name=True, including_default_value_fields=False)
        if OmegaConf.is_config(arg):
            arg = OmegaConf.to_container(arg, resolve=True)
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


class SimpleExperimentTracker:
    def __init__(self, exp_tracker_cfg, experiment_id, run_id):
        self._experiment_id = experiment_id
        self._run_id = run_id
        self._log_params = exp_tracker_cfg.log_params
        self._log_metrics = exp_tracker_cfg.log_metrics

    def log_params(self, *args, **kwargs):
        if self._log_params:
            params_dict = make_dict(False, *args, **kwargs)
            log.info(f"[{self._experiment_id}/{self._run_id}] log params [{json.dumps(params_dict)}]")

    def log_metrics(self, step_timestamp, step_idx, **kwargs):
        if self._log_metrics:
            metrics_dict = make_dict(True, **kwargs)
            log.info(
                f"[{self._experiment_id}/{self._run_id}] log metrics at step #{step_idx} [{json.dumps(metrics_dict)}]"
            )

    def terminate_failure(self):
        pass

    def terminate_success(self):
        pass
