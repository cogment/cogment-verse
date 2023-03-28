import logging
import multiprocessing as mp
import os
import subprocess
import shutil

import hydra
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import cogment_verse
from cogment_verse.constants import CONFIG_DIR, DEFAULT_CONFIG_NAME, DEFAULT_WORK_DIR, FT_DIR, TEST_DIR
from cogment_verse.processes.popen_process import PopenProcess
from cogment_verse.services_directory import ServiceType
from local_services import launch_local_services

log = logging.getLogger(__name__)

TEST_WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cogment_verse"))

CONFIG_REL_PATH = os.path.relpath(CONFIG_DIR, os.path.abspath(os.path.dirname(__file__)))

TEST_CONFIG_PATH = os.path.join(os.path.dirname(__file__), ".config")
TEST_CONFIG_REL_PATH = os.path.relpath(TEST_CONFIG_PATH, os.path.abspath(os.path.dirname(__file__)))


@pytest.fixture(scope="module")
def prepare_config():

    # Create tmp .config dir if not existant
    # os.makedirs(TEST_CONFIG_PATH, exist_ok=True)

    # Copy config content to .config
    shutil.copytree(CONFIG_DIR, TEST_CONFIG_PATH, dirs_exist_ok=True)

    # Copy smoke test experiment config to .config
    shutil.copytree(
        os.path.join(FT_DIR, "test_config"), os.path.join(TEST_CONFIG_PATH, "experiment"), dirs_exist_ok=True
    )

    yield

    shutil.rmtree(TEST_CONFIG_PATH, ignore_errors=True)


def test_simple_dqn_cartpole(prepare_config):

    args = ["python", "-m", "tests.functional.test_platform", "+experiment=simple_dqn/cartpole_ft"]
    proc = subprocess.Popen(args=args)
    proc.communicate()  # Wait for subprocess to complete
    assert proc.returncode == 0


@hydra.main(version_base=None, config_path=TEST_CONFIG_REL_PATH, config_name=DEFAULT_CONFIG_NAME)
def main(cfg):
    work_dir = os.environ.get("COGMENT_VERSE_WORK_DIR", DEFAULT_WORK_DIR)
    app = cogment_verse.App(cfg, work_dir=work_dir)

    app.start()
    app.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
