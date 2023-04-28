
import multiprocessing
import subprocess
from sagemaker_containers import training_job_utils


def run_script(module_name, *args):
    subprocess.run(["python", "-m", module_name, *args])


def main():
    hyperparameters = training_job_utils.get_hyperparameters()
    main_args = str(hyperparameters["main_args"])
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
