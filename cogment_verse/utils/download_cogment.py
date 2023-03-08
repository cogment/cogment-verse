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

from enum import Enum
from tempfile import mkdtemp
import json
import logging
import os
import platform
import re
import stat
import subprocess
from urllib.request import urlretrieve, urlopen

from cogment.errors import CogmentError

VERSION_NUMBER_RE = re.compile(r"[0-9]+.[0-9]+.[0-9]+(?:-[a-zA-Z0-9]+)?")

log = logging.getLogger(__name__)


class Arch(Enum):
    AMD64 = "amd64"
    ARM64 = "arm64"


def get_current_arch():
    py_machine = platform.machine()
    if py_machine in ["x86_64", "i686", "AMD64"]:
        return Arch.AMD64

    if py_machine in ["arm64"]:
        return Arch.ARM64

    raise CogmentError(f"Unsupported architecture [{py_machine}]")


class Os(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"


def get_current_os():
    py_system = platform.system()
    if py_system in ["Darwin"]:
        return Os.MACOS
    if py_system in ["Windows"]:
        return Os.WINDOWS
    if py_system in ["Linux"]:
        return Os.LINUX

    raise CogmentError(f"Unsupported os [{py_system}]")


def get_latest_release_version():
    with urlopen("https://api.github.com/repos/cogment/cogment/releases/latest") as res:
        parsed_body = json.load(res)

    return parsed_body["tag_name"]


def download_cogment(output_dir=None, desired_version=None, desired_arch=None, desired_os=None, force_download=False):
    """
    Download a version of cogment

    Parameters:
    - output_dir (string, optional): the output directory, if undefined a temporary directory will be used.
    - desired_version (string, optional): the desired version,
      if undefined the latest released version (excluding prereleases) will be used.
    - desired_arch (Arch, optional): the desired architecture,
      if undefined the current architecture will be detected and used.
    - os (Os, optional): the desired os, if undefined the current os will be detected and used.

    Returns:
        path to the downloaded cogment
    """
    if not output_dir:
        output_dir = mkdtemp()
    else:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if not desired_version:
        desired_version = get_latest_release_version()

    try:
        desired_version = VERSION_NUMBER_RE.findall(desired_version)[0]
    except Exception as error:
        raise CogmentError(
            f"Desired cogment version [{desired_version}] doesn't follow the expected patterns"
        ) from error

    if not desired_arch:
        desired_arch = get_current_arch()

    if not desired_os:
        desired_os = get_current_os()

    cogment_url = (
        "https://github.com/cogment/cogment/releases/download/"
        + f"v{desired_version}/cogment-{desired_os.value}-{desired_arch.value}"
    )

    cogment_filename = os.path.join(output_dir, "cogment")
    if desired_os == Os.WINDOWS:
        cogment_url += ".exe"
        cogment_filename += ".exe"

    if not force_download:
        try:
            res = subprocess.run([cogment_filename, "version"], capture_output=True, check=True)
            current_version = VERSION_NUMBER_RE.findall(res.stdout.decode("utf-8"))[0]
            if current_version == desired_version:
                # The current version of cogment matches the desired version
                return cogment_filename
        except OSError:
            # Unable to execute cogment, maybe it's missing or maybe it's corrupted let's just override it with a new one
            pass

    try:
        log.info(f"Downloading Cogment [{desired_version}] from [{cogment_url}]")
        cogment_filename, _ = urlretrieve(cogment_url, cogment_filename)
    except Exception as error:
        raise CogmentError(
            f"Unable to retrieve cogment version [{desired_version}] for arch "
            + f"[{desired_arch}] and os [{desired_os}] from [{cogment_url}] to [{cogment_filename}]"
        ) from error

    # Make sure it is executable
    cogment_stat = os.stat(cogment_filename)
    os.chmod(cogment_filename, cogment_stat.st_mode | stat.S_IEXEC)

    return cogment_filename
