import json
import multiprocessing
import os
import subprocess

# PREFIX = "./"
PREFIX = "/opt/ml/"  # Sagemaker's required prefix
PARAM_PATH = os.path.join(PREFIX, "input/config/")


def run_script(module_name, *args):
    subprocess.run(["python", "-m", module_name, *args])


def main():
    PARAM_PATH = os.path.join(PREFIX, "input/config/")
    hyper_params_path = PARAM_PATH + "hyperparameters.json"
    with open(hyper_params_path, "r", encoding="utf-8") as params:
        hyper_params = json.load(params)
    main_args = str(hyper_params["main_args"])
    service_process = multiprocessing.Process(target=run_script, args=("simple_mlflow",))
    client_process = multiprocessing.Process(
        target=run_script,
        args=(
            "main",
            main_args,
        ),
    )

    service_process.start()
    client_process.start()

    service_process.join()
    client_process.join()


if __name__ == "__main__":
    main()
