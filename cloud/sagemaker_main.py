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
    upload_process = multiprocessing.Process(
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
    upload_process.start()
    upload_mlflow_process.start()

    # Check if the cog-verse is still running
    while upload_process.is_alive():
        if not cogverse_process.is_alive():
            mlflow_process.terminate()
            upload_process.terminate()
            upload_mlflow_process.terminate()
            break
        time.sleep(20)

    mlflow_process.join()
    cogverse_process.join()
    upload_process.join()
    upload_mlflow_process.join()

    # Terminate the subprocesses after they're done running
    mlflow_process.terminate()
    cogverse_process.terminate()
    upload_process.terminate()
    upload_mlflow_process.terminate()


def run_script(module_name, *args):
    """Execute Python script"""
    subprocess.run(["python", "-m", module_name, *args], check=True)


def upload_to_s3_periodically(bucket: str, repo: str, interval: int = 300):
    """Upload all data including in .cog_verse excepted `bin` folder"""
    cwd_dir = os.getcwd()
    project_dir = f"{cwd_dir}/.cogment_verse"
    archive_name = "model.tar.gz"
    ignore_folders = ["bin"]
    s3_key = f"{repo}/models/{archive_name}"

    while True:
        pack_archive(
            project_dir=project_dir,
            main_dir=cwd_dir,
            output_path=cwd_dir,
            archive_name=archive_name,
            source_dir_names=["mlflow", "model_registry"],
            ignore_folders=ignore_folders,
        )
        local_path = f"{cwd_dir}/{archive_name}"
        upload_to_s3(local_path=local_path, bucket=bucket, s3_key=s3_key)
        delete_archive(local_path)
        print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
        time.sleep(interval)


def upload_to_mlflow_db_s3_realtime(bucket: str, repo: str, interval: int = 5):
    """Upload mlflow data to S3"""
    cwd_dir = os.getcwd()
    project_dir = f"{cwd_dir}/.cogment_verse"
    archive_name = "mlflow_db.tar.gz"

    while True:
        try:
            pack_archive(
                project_dir=project_dir,
                main_dir=cwd_dir,
                output_path=cwd_dir,
                source_dir_names=["mlflow"],
                archive_name=archive_name,
            )
            local_path = f"{cwd_dir}/{archive_name}"
            upload_to_s3(local_path=local_path, bucket=bucket, s3_key=f"{repo}/mlflow/{archive_name}")
            delete_archive(local_path)
        except ValueError as err_msg:
            print(f"Error: {err_msg}")
            continue
        time.sleep(interval)


if __name__ == "__main__":
    main()
