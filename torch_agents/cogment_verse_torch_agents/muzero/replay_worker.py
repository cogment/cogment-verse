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


class ReplayBufferWorker(mp.Process):
    def __init__(
        self,
        sample_queue,
        batch_queue,
        reanalyze_queue,
        reanalyze_update_queue,
        config,
        reward_distribution,
        value_distribution,
    ):
        super().__init__()
        self._sample_queue = sample_queue
        self._batch_queue = batch_queue
        self._reanalyze_queue = reanalyze_queue
        self._reanalyze_update_queue = reanalyze_update_queue
        self._replay_buffer_size = mp.Value(ctypes.c_uint32, 0)
        self._training_config = config
        self._device = config.train_device
        self.reward_distribution = reward_distribution
        self.value_distribution = value_distribution

    def run(self):
        torch.set_num_threads(self._training_config.threads_per_worker)
        episode_samples = {}
        replay_buffer = TrialReplayBuffer(
            max_size=self._training_config.max_replay_buffer_size,
            discount_rate=self._training_config.discount_rate,
            bootstrap_steps=self._training_config.bootstrap_steps,
        )

        zero_reward_probs = self.reward_distribution.compute_target(torch.tensor(0.0)).cpu().detach()
        zero_value_probs = self.value_distribution.compute_target(torch.tensor(0.0)).cpu().detach()

        while True:
            # Add any queued data to the replay buffer
            try:
                trial_id, sample = self._sample_queue.get_nowait()

                if trial_id not in episode_samples:
                    episode_samples[trial_id] = Episode(
                        sample.state,
                        self._training_config.discount_rate,
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
                        self._training_config.bootstrap_steps, self._training_config.discount_rate
                    )
                    replay_buffer.update_episode(episode_samples.pop(trial_id))
            except queue.Empty:
                pass

            self._replay_buffer_size.value = replay_buffer.size()
            if self._replay_buffer_size.value < self._training_config.min_replay_buffer_size:
                continue

            # Fetch/perform any pending reanalyze updates
            if not self._reanalyze_update_queue.empty():
                episode_id, episode = self._reanalyze_update_queue.get()
                # torch multiprocessing issue: need to create a process-local copy
                replay_buffer.update_episode(episode.clone(), key=episode_id)
                del episode

            # Queue next reanalyze update
            if not self._reanalyze_queue.full():
                try:
                    # testing, sampling strategy
                    keys = list(replay_buffer.episodes.keys())
                    p = torch.tensor([replay_buffer.episodes[key].timestamp for key in keys], dtype=torch.double)
                    p -= p.min() - 0.1
                    p /= p.sum()
                    dist = torch.distributions.Categorical(p)
                    # episode_id = np.random.randint(0, len(replay_buffer.episodes))
                    key_id = dist.sample().item()
                    self._reanalyze_queue.put((keys[key_id], replay_buffer.episodes[keys[key_id]]), timeout=1.0)
                except queue.Full:
                    pass

                del p
                del dist

            # Sample a batch and add it to the training queue
            if replay_buffer.size() >= self._training_config.min_replay_buffer_size and not self._batch_queue.full():
                batch = replay_buffer.sample(self._training_config.rollout_length, self._training_config.batch_size)
                for item in batch:
                    item.to(self._training_config.train_device)
                try:
                    self._batch_queue.put(EpisodeBatch(*batch), timeout=1.0)
                except queue.Full:
                    pass

    def add_sample(self, trial_id, sample):
        self._sample_queue.put((trial_id, sample))

    def size(self):
        return self._replay_buffer_size.value
