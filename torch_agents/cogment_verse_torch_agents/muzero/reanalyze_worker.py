# Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import ctypes
import io
import queue
import torch
import torch.multiprocessing as mp

from cogment_verse.utils import LRU
from cogment_verse_torch_agents.muzero.replay_buffer import Episode
from cogment_verse_torch_agents.muzero.agent import MuZeroAgent
from cogment_verse_torch_agents.muzero.utils import MuZeroWorker, flush_queue


class ReanalyzeWorker(MuZeroWorker):
    def __init__(
        self,
        reanalyze_queue,
        reanalyze_update_queue,
        model_id,
        reward_distribution,
        value_distribution,
        config,
        manager,
    ):
        super().__init__(config, manager)
        self._agent_queue = manager.Queue(2)
        self._reanalyze_queue = reanalyze_queue
        self._reanalyze_update_queue = reanalyze_update_queue
        self._device = config.reanalyze_device
        self._reanalyzed_samples = mp.Value(ctypes.c_uint64, 0)
        self.reward_distribution = reward_distribution
        self.value_distribution = value_distribution
        self._model_cache = LRU(1)
        self._model_id = model_id

    def reanalyzed_samples(self):
        return self._reanalyzed_samples.value

    def update_agent(self, agent):
        assert not self.done.value
        self._agent_queue.put(agent.serialize_to_buffer())

    @torch.no_grad()
    async def main(self):
        agent = MuZeroAgent.load(io.BytesIO(self._agent_queue.get()), "cpu")
        agent.set_device(self._device)

        while not self.done.value:
            try:
                agent = MuZeroAgent.load(io.BytesIO(self._agent_queue.get_nowait()), "cpu")
                agent.set_device(self._device)
            except queue.Empty:
                pass

            try:
                episode_id, episode = self._reanalyze_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            reanalyze_episode = Episode(
                episode.states[0],
                agent.params.training.discount_rate,
                zero_reward_probs=episode.zero_reward_probs,
                zero_value_probs=episode.zero_value_probs,
            )
            for step in range(len(episode)):
                policy, _, value = agent.reanalyze(episode.states[step])
                reward_probs = self.reward_distribution.compute_target(torch.tensor(episode.rewards[step])).cpu()
                value_probs = self.value_distribution.compute_target(value).cpu()
                reanalyze_episode.add_step(
                    episode.states[step + 1],
                    episode.actions[step],
                    reward_probs,
                    episode.rewards[step],
                    episode.done[step],
                    policy,
                    value_probs,
                    value.detach().cpu().item(),
                )
            del episode
            reanalyze_episode.bootstrap_value(
                agent.params.training.bootstrap_steps, agent.params.training.discount_rate
            )
            self._reanalyze_update_queue.put((episode_id, reanalyze_episode))
            self._reanalyzed_samples.value += len(reanalyze_episode)

    def cleanup(self):
        flush_queue(self._reanalyze_update_queue)
        flush_queue(self._agent_queue)
