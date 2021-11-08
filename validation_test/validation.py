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


def main():

    experiments = [
        "cartpole_rainbow",
        "cartpole_dqn",
        "lander_rainbow_demo",
        "pendulum_td3",
        "pendulum_ddpg",
        "lander_continuous_td3",
        "lander_continuous_ddpg",
        "walker_td3",
        "walker_ddpg",
    ]

    experiment_names = {}

    for experiment in experiments:
        timestamp = time.time()
        run_name = experiment + "_val_test_" + str(timestamp)
        experiment_names[experiment] = run_name
        launch_test = "docker-compose run client start " + experiment + " --run_id " + run_name
        os.system(launch_test)

    exp_str = " --exp "
    names_str = " --names "
    for exp, name in experiment_names.items():
        exp_str += f" {exp} "
        names_str += f" {name} "

    launch_checker = f"docker-compose run validation_test python main.py {exp_str} {names_str}"
    print(launch_checker)
    exit_code = os.system(launch_checker)
    print(exit_code)
    assert exit_code == 0


if __name__ == "__main__":
    main()
