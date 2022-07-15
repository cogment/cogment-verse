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
import queue
import torch
import torch.multiprocessing as mp

from cogment_verse_torch_agents.muzero.replay_buffer import Episode, TrialReplayBuffer, EpisodeBatch
from cogment_verse_torch_agents.muzero.utils import MuZeroWorker, flush_queue


class ReplayBufferWorker(MuZeroWorker):
    def __init__(
        self,
        batch_queue,
        config,
        reward_distribution,
        value_distribution,
        manager,
    ):
        super().__init__(config, manager)
        self._sample_queue = manager.Queue()
        self.reanalyze_update_queue = manager.Queue(config.reanalyze_workers + 1)
        self.reanalyze_queue = manager.Queue(config.reanalyze_workers + 1)
        self._batch_queue = batch_queue
        self._replay_buffer_size = mp.Value(ctypes.c_uint32, 0)
        self.reward_distribution = reward_distribution
        self.value_distribution = value_distribution

    async def main(self):
        episode_samples = {}
        replay_buffer = TrialReplayBuffer(max_size=self.config.training.max_replay_buffer_size)

        zero_reward_probs = self.reward_distribution.compute_target(torch.tensor(0.0)).cpu().detach()
        zero_value_probs = self.value_distribution.compute_target(torch.tensor(0.0)).cpu().detach()

        while not self.done.value:
            # Add any queued data to the replay buffer
            try:
                trial_id, sample = self._sample_queue.get_nowait()

                if trial_id not in episode_samples:
                    episode_samples[trial_id] = Episode(
                        sample.state,
                        self.config.training.discount_rate,
                        zero_reward_probs=zero_reward_probs,
                        zero_value_probs=zero_value_probs,
                    )

                with torch.no_grad():
                    reward_probs = self.reward_distribution.compute_target(torch.tensor(sample.reward)).cpu()
                    value_probs = self.value_distribution.compute_target(torch.tensor(sample.value)).cpu()

                episode_samples[trial_id].add_step(
                    sample.next_state,
                    sample.action,
                    reward_probs,
                    sample.reward,
                    sample.done,
                    sample.policy,
                    value_probs,
                    sample.value,
                )

                if sample.done:
                    episode_samples[trial_id].bootstrap_value(
                        self.config.training.bootstrap_steps, self.config.training.discount_rate
                    )
                    replay_buffer.update_episode(episode_samples.pop(trial_id))
            except queue.Empty:
                pass

            self._replay_buffer_size.value = replay_buffer.size()
            if self._replay_buffer_size.value < self.config.training.min_replay_buffer_size:
                continue

            # Fetch/perform any pending reanalyze updates
            if not self.reanalyze_update_queue.empty():
                episode_id, episode = self.reanalyze_update_queue.get()
                # torch multiprocessing issue: need to create a process-local copy
                replay_buffer.update_episode(episode.clone(), key=episode_id)
                del episode

            # Queue next reanalyze update
            if not self.reanalyze_queue.full():
                # Don't just reanalyze the oldest episodes since these are most likely
                # to be ejected when the replay buffer is full. Instead we sample randomly
                # with a probability weighted by episode "staleness"
                keys = list(replay_buffer.episodes.keys())
                probs = torch.tensor([replay_buffer.episodes[key].timestamp for key in keys], dtype=torch.double)
                probs -= probs.min() - 0.1
                probs /= probs.sum()
                dist = torch.distributions.Categorical(probs)
                key_id = dist.sample().item()
                self.reanalyze_queue.put_nowait((keys[key_id], replay_buffer.episodes[keys[key_id]]))
                del probs
                del dist

            # Sample a batch and add it to the training queue
            if replay_buffer.size() >= self.config.training.min_replay_buffer_size and not self._batch_queue.full():
                batch = replay_buffer.sample(self.config.mcts.rollout_length, self.config.training.batch_size)
                for item in batch:
                    item.to(self.config.train_device)
                self._batch_queue.put_nowait(EpisodeBatch(*batch))

    def cleanup(self):
        # Consume remaining items
        flush_queue(self._sample_queue)
        flush_queue(self.reanalyze_queue)
        flush_queue(self._batch_queue)

        # Note: we do _not_ flush the reanalyze update queue since this should
        # be done by the reanalyze worker processes due to the way torch MP works

    def add_sample(self, trial_id, sample):
        assert not self.done.value
        self._sample_queue.put((trial_id, sample))

    def size(self):
        return self._replay_buffer_size.value
