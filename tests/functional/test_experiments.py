import logging
import multiprocessing as mp
import os
import shutil
import subprocess

import hydra
import pytest

import cogment_verse
from cogment_verse.constants import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_NAME, FUNCTIONAL_TEST_DIR

log = logging.getLogger(__name__)

TEST_WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".tmp_cogment_verse"))

CONFIG_REL_PATH = os.path.relpath(DEFAULT_CONFIG_DIR, os.path.abspath(os.path.dirname(__file__)))

TEST_CONFIG_PATH = os.path.join(os.path.dirname(__file__), ".tmp_config")
TEST_CONFIG_REL_PATH = os.path.relpath(TEST_CONFIG_PATH, os.path.abspath(os.path.dirname(__file__)))

DEFAULT_TEST_TIMEOUT = 60  # seconds

TEST_EXPERIMENTS = [
    "random_cartpole_ft",
    "ppo/hopper_ft",
    "ppo/lunar_lander_continuous_ft",
    "simple_dqn/cartpole_ft",
    # "simple_a2c/ant_ft", # requires: pip install -r isaac_requirements.txt
    "simple_a2c/cartpole_ft",
    "td3/lunar_lander_continuous_ft",
]


@pytest.fixture(name="_prepare_config", scope="module")
def fixture_prepare_config():
    """ Fixture used to setup hydra configs for functional tests.
        Copies original configs together with functional test configs in a tmp folder.
        The teardown removes the configs and the tmp work directory.
    """

    # Copy config content to .tmp_config
    shutil.copytree(DEFAULT_CONFIG_DIR, TEST_CONFIG_PATH, dirs_exist_ok=True)
    # Copy smoke test experiment config to .tmp_config
    shutil.copytree(
        os.path.join(FUNCTIONAL_TEST_DIR, "test_config"), os.path.join(TEST_CONFIG_PATH, "experiment"), dirs_exist_ok=True
    )

    yield
    # clean tmp folders
    shutil.rmtree(TEST_CONFIG_PATH, ignore_errors=True)
    shutil.rmtree(TEST_WORK_DIR, ignore_errors=True)


@pytest.mark.functional
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
def test_default_experiment(_prepare_config):
    """ Test the default config."""
    proc = subprocess.Popen(
        args=[
            "python",
            "-m",
            "tests.functional.test_experiments",
            "run=headless_play",
            "services/experiment_tracker@run.experiment_tracker=simple",
        ]
    )
    proc.communicate()
    assert proc.returncode == 0


@pytest.mark.functional
@pytest.mark.parametrize("experiment", TEST_EXPERIMENTS)
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
def test_experiment(_prepare_config, experiment):
    """ Test individual experiments."""
    proc = subprocess.Popen(args=["python", "-m", "tests.functional.test_experiments", f"+experiment={experiment}"])
    proc.communicate()
    assert proc.returncode == 0


@pytest.mark.functional
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
def test__model_registry(_prepare_config):
    """ Test that a trained model is properly archived to the model registry. 
        Then, test that the same model can be retrieved to play trials.
    """
    proc = subprocess.Popen(
        args=["python", "-m", "tests.functional.test_experiments", "+experiment=simple_dqn/connect_four_ft"]
    )
    proc.communicate()
    assert proc.returncode == 0

    proc = subprocess.Popen(
        args=["python", "-m", "tests.functional.test_experiments", "+experiment=simple_dqn/observe_connect_four_ft"]
    )
    proc.communicate()
    assert proc.returncode == 0


@hydra.main(version_base=None, config_path=TEST_CONFIG_REL_PATH, config_name=DEFAULT_CONFIG_NAME)
def main(cfg):
    work_dir = os.environ.get("COGMENT_VERSE_WORK_DIR", TEST_WORK_DIR)
    app = cogment_verse.App(cfg, work_dir=work_dir)

    app.start()
    app.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
