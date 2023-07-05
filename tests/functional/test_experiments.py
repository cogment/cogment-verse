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

# pylint: disable=consider-using-with

import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys

import hydra
import pytest

import cogment_verse
from cogment_verse.constants import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_NAME, FUNCTIONAL_TEST_DIR

log = logging.getLogger(__name__)

TEST_WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".tmp_cogment_verse"))
CONFIG_REL_PATH = os.path.relpath(DEFAULT_CONFIG_DIR, os.path.abspath(os.path.dirname(__file__)))
TEST_CONFIG_PATH = os.path.join(os.path.dirname(__file__), ".tmp_config")
TEST_CONFIG_REL_PATH = os.path.relpath(TEST_CONFIG_PATH, os.path.abspath(os.path.dirname(__file__)))
DEFAULT_TEST_TIMEOUT = 500  # seconds

TEST_EXPERIMENTS = [
    "random_cartpole_ft",
    "ppo/hopper_ft",
    "ppo/lunar_lander_continuous_ft",
    "sac/hopper_ft",
    "simple_dqn/cartpole_ft",
    # "simple_a2c/ant_ft", # requires: pip install -r isaac_requirements.txt
    "simple_a2c/cartpole_ft",
    "td3/lunar_lander_continuous_ft",
]


@pytest.fixture(name="_prepare_config", scope="module")
def fixture_prepare_config():
    """Fixture used to setup hydra configs for functional tests.
    Copies original configs together with functional test configs in a tmp folder.
    The teardown removes the configs and the tmp work directory.
    """

    # Copy config content to .tmp_config
    shutil.copytree(DEFAULT_CONFIG_DIR, TEST_CONFIG_PATH, dirs_exist_ok=True)
    # Copy smoke test experiment config to .tmp_config
    shutil.copytree(
        os.path.join(FUNCTIONAL_TEST_DIR, "test_config"),
        os.path.join(TEST_CONFIG_PATH, "experiment"),
        dirs_exist_ok=True,
    )

    yield
    # clean tmp folders
    shutil.rmtree(TEST_CONFIG_PATH, ignore_errors=True)
    shutil.rmtree(TEST_WORK_DIR, ignore_errors=True)


def get_experiment_configs():
    """Retrieve the name of all experiment config files."""
    configs = []
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(os.path.join(DEFAULT_CONFIG_DIR, "experiment")):
        for file in files:
            # Add the file name to the list
            base_name, extension = os.path.splitext(file)
            configs.append(
                os.path.relpath(os.path.join(root, base_name), os.path.join(DEFAULT_CONFIG_DIR, "experiment"))
            )

    return configs


@pytest.mark.functional
@pytest.mark.parametrize("config", get_experiment_configs())
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
def test_hydra_composition(config):
    """Test that hydra configurations can be created for all experiments"""
    proc = subprocess.Popen(args=[sys.executable, "-m", "main", f"+experiment={config}", "--info", "defaults-tree"])
    proc.communicate()
    assert proc.returncode == 0


@pytest.mark.functional
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
def test_default_experiment(_prepare_config):
    """Test the default config."""
    proc = subprocess.Popen(
        args=[
            sys.executable,
            "-m",
            "tests.functional.test_experiments",
            "run=headless_play",
            "run/experiment_tracker=simple",
        ]
    )
    proc.communicate()
    assert proc.returncode == 0


@pytest.mark.functional
@pytest.mark.parametrize("experiment", TEST_EXPERIMENTS)
@pytest.mark.timeout(DEFAULT_TEST_TIMEOUT)
def test_experiment(_prepare_config, experiment):
    """Test individual experiments."""
    proc = subprocess.Popen(
        args=[sys.executable, "-m", "tests.functional.test_experiments", f"+experiment={experiment}"]
    )
    proc.communicate()
    assert proc.returncode == 0


@pytest.mark.functional
@pytest.mark.timeout(2 * DEFAULT_TEST_TIMEOUT)
def test__model_registry(_prepare_config):
    """Test that a trained model is properly archived to the model registry.
    Then, test that the same model can be retrieved to play trials.
    """
    proc = subprocess.Popen(
        args=[sys.executable, "-m", "tests.functional.test_experiments", "+experiment=simple_dqn/connect_four_ft"]
    )
    proc.communicate()
    assert proc.returncode == 0

    # Using a specific model iteration number
    proc = subprocess.Popen(
        args=[
            sys.executable,
            "-m",
            "tests.functional.test_experiments",
            "+experiment=simple_dqn/observe_connect_four_specific_iteration_ft",
        ]
    )
    proc.communicate()
    assert proc.returncode == 0

    # Using the model iteration -1
    proc = subprocess.Popen(
        args=[
            sys.executable,
            "-m",
            "tests.functional.test_experiments",
            "+experiment=simple_dqn/observe_connect_four_ft",
        ]
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
