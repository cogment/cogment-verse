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

import argparse
import os
from subprocess import check_call


def main():
    parser = argparse.ArgumentParser(description="Start a simple mlflow server")
    parser.add_argument("port", type=int, default=6969, nargs="?", help="TCP port (optional, default is 3000)")
    args = parser.parse_args()

    mlflow_data_dir = os.path.join(os.path.dirname(__file__), ".cogment_verse/mlflow")
    os.makedirs(mlflow_data_dir, exist_ok=True)
    check_call(
        args=[
            "mlflow",
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            f"{args.port}",
            "--backend-store-uri",
            f"sqlite:///{mlflow_data_dir}/mlflow.db",
            "--default-artifact-root",
            f"{mlflow_data_dir}",
        ]
    )


if __name__ == "__main__":
    main()
