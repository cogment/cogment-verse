import logging
import os
import subprocess
import time

import hydra
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf


import cogment_verse
from cogment_verse.constants import CONFIG_DIR, DEFAULT_WORK_DIR
from cogment_verse.processes.popen_process import PopenProcess
from cogment_verse.services_directory import ServiceType
from local_services import launch_local_services

log = logging.getLogger(__name__)

TEST_WORK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_WORK_DIR))


# @pytest.fixture(scope="module")
# def background_process():
#     # Start the background process
#     process = subprocess.Popen(["python", "my_background_process.py"])

#     # Wait for the process to start up
#     time.sleep(1)

#     # Yield the process object so that it can be used in the test functions
#     yield process

#     # Stop the background process
#     process.kill()


@pytest.fixture(scope='module')
def launch_platform():
    with initialize(version_base=None, config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config", overrides=["db=mysql", "db.user=me"])
        print(OmegaConf.to_yaml(cfg))

        #launch_local_services(cfg=cfg, work_dir=TEST_WORK_DIR)


def test_simple_dqn_cartpole(launch_platform):
    PopenProcess(
        name="test",
        args=["python", "-m", "main", "+experiment=simple_dqn/cartpole"]
    )

    # # Wait for the process to start up
    # time.sleep(1)


