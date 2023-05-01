import json
import multiprocessing
import os
import subprocess
import tarfile
from cloud.sagemaker_utils import upload_to_s3, pack_folder
import time

PREFIX = "/opt/ml/"  # Sagemaker's required prefix
PARAM_PATH = os.path.join(PREFIX, "input/config/")
SOURCE_ARCHIVE = os.path.join(PREFIX, "input", "data", "training", "source_code.tar.gz")
SOURCE_DIR = os.path.join(PREFIX, "code/")

with tarfile.open(SOURCE_ARCHIVE, "r:gz") as tar:
    tar.extractall(SOURCE_DIR)


def run_script(module_name, *args):
    subprocess.run(["python", "-m", module_name, *args])


def upload_to_s3_periodically(bucket: str, repo: str, interval: int = 300):
    archive_name = "model.tar.gz"
    ignore_list = [".cogment_verse/bin"]
    pack_folder(folder_path=".cogment_verse", archive_name=archive_name, ignore_list=ignore_list)

    while True:
        upload_to_s3(archive_name, bucket, f"{repo}/models/{archive_name}")
        time.sleep(interval)


def main():
    hyper_params_path = PARAM_PATH + "hyperparameters.json"
    with open(hyper_params_path, "r", encoding="utf-8") as params:
        hyper_params = json.load(params)
    main_args = str(hyper_params["main_args"])
    s3_bucket = str(hyper_params["s3_bucket"])
    repo = str(hyper_params["repo"])

    # Run the cogment verse
    mlflow_process = multiprocessing.Process(target=run_script, args=("simple_mlflow",))
    cogverse_process = multiprocessing.Process(
        target=run_script,
        args=(
            "main",
            main_args,
        ),
    )
    upload_process = multiprocessing.Process(
        target=upload_to_s3_periodically,
        args=(
            s3_bucket,
            repo,
            300,  # Interval in seconds
        ),
    )

    mlflow_process.start()
    cogverse_process.start()
    upload_process.start()

    # Check every 60 second
    while upload_process.is_alive():
        if not cogverse_process.is_alive():
            upload_process.terminate()
            mlflow_process.terminate()
            break
        time.sleep(60)

    mlflow_process.join()
    cogverse_process.join()
    upload_process.join()

    # Terminate the subprocesses after they're done running
    mlflow_process.terminate()
    cogverse_process.terminate()
    upload_process.terminate()


if __name__ == "__main__":
    main()
