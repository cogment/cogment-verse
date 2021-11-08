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

import time
import os
import sys

from mlflow.tracking import MlflowClient

import argparse


def parse_experiment():

    # defined command line options
    # this also generates --help and error handling
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--exp",
        nargs="*",
        type=str,
        default=[],
    )
    cli.add_argument(
        "--names",
        nargs="*",
        type=str,
        default=[],
    )

    args = cli.parse_args()

    experiment_names = {}
    experiment_count = min(len(args.exp), len(args.names))
    for i in range(experiment_count):
        experiment_names[args.exp[i]] = args.names[i]

    return experiment_names


class ExperimentRun:
    def __init__(self, name, experiment_name, reward):
        self.name = name
        self.experiment_name = experiment_name
        self.expected_reward = reward
        self.in_progress = True
        self.succed = False
        self.run_id = None
        self.max_reward_acheived = sys.float_info.min
        self.status = "None"

    def __str__(self):
        text = (
            self.experiment_name
            + "\ttest name:\t"
            + self.name
            + "\texpected:\t"
            + str(self.expected_reward)
            + "\trewards and got\t"
            + str(self.max_reward_acheived)
            + "\tstatus: "
            + self.status
        )
        if self.succed:
            text += "\t-> TEST SUCCEED"
        else:
            text += "\t> TEST FAILED"
        return text


def main():

    experiment_names = parse_experiment()

    expected_total_rewards = {
        "cartpole_rainbow": 100,
        "cartpole_dqn": 100,
        "lander_rainbow_demo": 100,
        "pendulum_td3": -10.0,
        "pendulum_ddpg": -10.0,
        "lander_continuous_td3": 100,
        "lander_continuous_ddpg": 100,
        "walker_td3": 100,
        "walker_ddpg": 100,
    }

    experiment_runs = {}
    for experiment, run_name in experiment_names.items():
        reward = expected_total_rewards.get(experiment, 100)
        experiment_runs[run_name] = ExperimentRun(run_name, experiment, reward)

    stating_time = time.time()
    time_out = 60 * 60 * 6  # 6 hours

    hostname = "mlflow"
    port = 5000

    just_url = "http://" + hostname + ":" + str(port)
    os.environ["MLFLOW_TRACKING_URI"] = just_url

    client = MlflowClient()

    in_progress = True
    # pylint: disable=too-many-nested-blocks
    while in_progress and time.time() - stating_time < time_out:
        in_progress = False

        for run_name, experiment_run in experiment_runs.items():
            if experiment_run.run_id is None:
                experiment = client.get_experiment_by_name("/" + experiment_run.experiment_name)
                if experiment is not None:
                    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
                    for run in runs:
                        run_dict = run.to_dictionary()
                        if "data" in run_dict and "mlflow.runName" in run_dict["data"]["tags"]:
                            data = run_dict["data"]
                            name = data["tags"]["mlflow.runName"]
                            if name == experiment_run.name:
                                info = run_dict["info"]
                                experiment_run.run_id = info["run_id"]
                                print("Found ", name, " with id ", experiment_run.run_id)
                                break
            elif experiment_run.in_progress:
                run = client.get_run(experiment_run.run_id)
                run_dict = run.to_dictionary()
                assert "info" in run_dict
                info = run_dict["info"]
                if "status" in info:
                    experiment_run.status = info["status"]
                    experiment_run.in_progress = experiment_run.in_progress and (info["status"] == "RUNNING")

                data = run_dict["data"]
                assert "metrics" in data
                metrics = data["metrics"]
                if "trial_total_reward" in metrics:
                    reward = metrics["trial_total_reward"]
                    if reward > experiment_run.expected_reward:
                        experiment_run.succed = True
                        experiment_run.in_progress = False
                    if reward > experiment_run.max_reward_acheived:
                        experiment_run.max_reward_acheived = reward

            in_progress = in_progress or experiment_run.in_progress
        time.sleep(10.0)

    for _, experiment_run in experiment_runs.items():
        print(experiment_run)

    print("Tests executed in ", time.time() - stating_time, " seconds")

    for _, experiment_run in experiment_runs.items():
        assert experiment_run.succed


if __name__ == "__main__":
    main()
