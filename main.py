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

import profile
import pstats
import logging
import os

import hydra
import torch.multiprocessing as mp

import cogment_verse

log = logging.getLogger(__name__)

# pylint: disable=C0209
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cogment_verse"))
    app = cogment_verse.App(cfg, work_dir=work_dir)

    app.start()
    app.join()


def run_profile():
    filename = []
    for i in range(1):
        filename = "profile_stats_%d.stats" % i
        profile.run("main()", filename)
    # Read all 5 stats files into a single object
    stats = pstats.Stats("profile_stats_0.stats")
    for i in range(1):
        stats.add("profile_stats_%d.stats" % i)

    # Clean up filenames for the report
    stats.strip_dirs()

    # Sort the statistics by the cumulative time spent in the function
    stats.sort_stats("tottime")

    stats.print_stats(50)


if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")
    main()
