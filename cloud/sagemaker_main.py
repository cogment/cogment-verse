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

import json
import multiprocessing
import os
import subprocess
import tarfile
import time

from cloud.sagemaker_utils import delete_archive, pack_archive, upload_to_s3

PREFIX = "/opt/ml/"  # Sagemaker's required prefix
PARAM_PATH = os.path.join(PREFIX, "input/config/")
SOURCE_ARCHIVE = os.path.join(PREFIX, "input", "data", "training", "source_code.tar.gz")
SOURCE_DIR = os.path.join(PREFIX, "code/")


def main():
    """API for runing cogment verse on Sagemaker"""
    # Create cogment verse folder.
    cwd_dir = os.getcwd()
    cog_verse = f"{cwd_dir}/.cogment_verse"
    mlflow_folder = f"{cog_verse}/mlflow"
    mr_folder = f"{cog_verse}/model_registry"
    os.mkdir(cog_verse)
    os.mkdir(mlflow_folder)
    os.mkdir(mr_folder)

    # Extract the source code to the main directory
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as tar:
        tar.extractall(SOURCE_DIR)

    # Retrieve hyperparameters
    hyper_params_path = PARAM_PATH + "hyperparameters.json"
    with open(hyper_params_path, "r", encoding="utf-8") as params:
        hyper_params = json.load(params)
    main_args = str(hyper_params["main_args"])
    s3_bucket = str(hyper_params["s3_bucket"])
    repo = str(hyper_params["repo"])

    # Define the cogment verse process
    mlflow_process = multiprocessing.Process(target=run_script, args=("simple_mlflow",))
    cogverse_process = multiprocessing.Process(
        target=run_script,
        args=(
            "main",
            main_args,
        ),
    )

    # Define process for uploading model to S3
    upload_model_process = multiprocessing.Process(
        target=upload_to_s3_periodically,
        args=(
            s3_bucket,
            repo,
            300,  # Interval in seconds
        ),
    )

    # Define process for uploading mlflow data for real-time tracking
    upload_mlflow_process = multiprocessing.Process(
        target=upload_to_mlflow_db_s3_realtime,
        args=(
            s3_bucket,
            repo,
        ),
    )

    # Start all processes
    mlflow_process.start()
    cogverse_process.start()
    upload_model_process.start()
    upload_mlflow_process.start()

    while True:
        time.sleep(10)
        if not cogverse_process.is_alive():
            # Terminate the subprocesses after they're done running
            mlflow_process.terminate()
            cogverse_process.terminate()
            upload_model_process.terminate()
            upload_mlflow_process.terminate()

            mlflow_process.join()
            cogverse_process.join()
            upload_model_process.join()
            upload_mlflow_process.join()
            break


def run_script(module_name, *args):
    """Execute Python script"""
    subprocess.run(["python", "-m", module_name, *args], check=True)


def upload_to_s3_periodically(bucket: str, repo: str, interval: int = 300):
    """Upload all data including in .cog_verse excepted `bin` folder"""
    cwd_dir = os.getcwd()
    project_dir = f"{cwd_dir}/.cogment_verse"
    archive_name = "model.tar.gz"
    s3_key = f"{repo}/models/{archive_name}"
    local_path = f"{cwd_dir}/{archive_name}"

    while True:
        try:
            pack_archive(
                project_dir=project_dir,
                main_dir=cwd_dir,
                output_path=cwd_dir,
                archive_name=archive_name,
                source_dir_names=["mlflow", "model_registry"],
            )
            upload_to_s3(local_path=local_path, bucket=bucket, s3_key=s3_key)
            delete_archive(local_path)
        except FileNotFoundError as err_msg:
            print(f"Error: {err_msg}")
            continue
        time.sleep(interval)


def upload_to_mlflow_db_s3_realtime(bucket: str, repo: str, interval: int = 5):
    """Upload mlflow data to S3"""
    cwd_dir = os.getcwd()
    project_dir = f"{cwd_dir}/.cogment_verse"
    archive_name = "mlflow_db.tar.gz"
    local_path = f"{cwd_dir}/{archive_name}"
    s3_key = f"{repo}/mlflow/{archive_name}"

    while True:
        try:
            pack_archive(
                project_dir=project_dir,
                main_dir=cwd_dir,
                output_path=cwd_dir,
                source_dir_names=["mlflow"],
                archive_name=archive_name,
            )
            upload_to_s3(local_path=local_path, bucket=bucket, s3_key=s3_key)
            delete_archive(local_path)
        except FileNotFoundError as err_msg:
            print(f"Error: {err_msg}")
            continue
        time.sleep(interval)


if __name__ == "__main__":
    main()
