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

import asyncio
import logging

log = logging.getLogger(__name__)


class TrialSample:
    def __init__(self, sample_pb, actor_classes):
        self._sample_pb = sample_pb
        self._actor_classes = actor_classes

    def _get_payload(self, payload_idx, pb_message_class=None, default=None):
        if payload_idx is None:
            return default

        payload = self._sample_pb.payloads[payload_idx]

        if pb_message_class is None:
            return payload

        message = pb_message_class()
        message.ParseFromString(payload)
        return message

    def get_trial_id(self):
        return self._sample_pb.trial_id

    def get_user_id(self):
        return self._sample_pb.user_id

    def get_tick_id(self):
        return self._sample_pb.tick_id

    def get_timestamp(self):
        return self._sample_pb.timestamp

    def get_trial_state(self):
        return self._sample_pb.state

    def _get_actor(self, actor_idx):
        actor = self._sample_pb.actor_samples[actor_idx]
        assert actor_idx == actor.actor

        return actor

    def count_actors(self):
        return len(self._sample_pb.actor_samples)

    def get_actor_observation(self, actor_idx, deserialize=True, default=None):
        actor = self._get_actor(actor_idx)
        return self._get_payload(
            actor.observation,
            pb_message_class=self._actor_classes[actor_idx].observation_space if deserialize else None,
            default=default,
        )

    def get_actor_action(self, actor_idx, deserialize=True, default=None):
        actor = self._get_actor(actor_idx)
        return self._get_payload(
            actor.action,
            pb_message_class=self._actor_classes[actor_idx].action_space if deserialize else None,
            default=default,
        )

    def get_actor_reward(self, actor_idx, default=None):
        actor = self._get_actor(actor_idx)
        reward = actor.reward
        if reward is None:
            return default

        return reward

    def get_actor_received_rewards(self, actor_idx):
        actor = self._get_actor(actor_idx)
        return [
            (reward.sender, reward.reward, reward.confidence, self._get_payload(reward.user_data))
            for reward in actor.received_rewards
        ]

    def get_actor_sent_rewards(self, actor_idx):
        actor = self._get_actor(actor_idx)
        return [
            (reward.receiver, reward.reward, reward.confidence, self._get_payload(reward.user_data))
            for reward in actor.sent_rewards
        ]

    def get_actor_received_messages(self, actor_idx):
        actor = self._get_actor(actor_idx)
        return [(msg.sender, self._get_payload(msg.payload)) for msg in actor.received_rewards]

    def get_actor_sent_messages(self, actor_idx):
        actor = self._get_actor(actor_idx)
        return [(msg.receiver, self._get_payload(msg.payload)) for msg in actor.sent_messages]


class RunSampleProducerSession:
    def __init__(
        self,
        cog_settings,
        run_id,
        trial_id,
        trial_params,
        produce_training_sample,
        run_config,
        run_sample_producer_impl,
    ):
        self.run_id = run_id
        self.trial_id = trial_id
        self._trial_params = trial_params
        self._actor_classes = [
            cog_settings.actor_classes[actor_params.actor_class] for actor_params in trial_params.actors
        ]
        self._trial_config_class = cog_settings.trial.config_type
        self._run_sample_producer_impl = run_sample_producer_impl
        self._produce_training_sample = produce_training_sample
        self._current_tick_id = 0

        self.run_config = run_config

        self._queue = asyncio.Queue(maxsize=10)

    # TODO Expose further helper functions to avoid the need to access directly _trial_params as needed
    def count_actors(self):
        return len(self._trial_params.actors)

    def get_trial_config(self, deserialize=True):
        raw_trial_config = self._trial_params.trial_config.content
        if not deserialize:
            return raw_trial_config

        trial_config = self._trial_config_class()
        trial_config.ParseFromString(raw_trial_config)
        return trial_config

    def exec(self):
        async def exec_run():
            log.debug(f"[{self.run_id}/{self.trial_id}] Starting sample producer...")
            impl_task = asyncio.create_task(self._run_sample_producer_impl(self))
            try:
                await impl_task
                log.debug(f"[{self.run_id}/{self.trial_id}] Sample producer succeeded")
            except asyncio.CancelledError:
                log.debug(f"[{self.run_id}/{self.trial_id}] Terminating sample producer")
                try:
                    await impl_task
                except asyncio.CancelledError:
                    pass
                log.debug(f"[{self.run_id}/{self.trial_id}] Sample producer terminated")
                raise
            except Exception as error:
                log.error(
                    f"[{self.run_id}/{self.trial_id}] Uncaught error occured during the sample production",
                    exc_info=error,
                )
                raise error

        self._task = asyncio.create_task(exec_run())
        return self._task

    async def on_trial_sample(self, sample):
        await self._queue.put(sample)

    async def on_trial_done(self):
        await self._queue.put(True)

    async def get_all_samples(self):
        while True:
            enqeued_item = await self._queue.get()
            if enqeued_item is True:
                # Trial done
                return
            trial_sample = TrialSample(enqeued_item, self._actor_classes)
            self._current_tick_id = trial_sample.get_tick_id()
            log.debug(f"[{self.run_id}] retrieving a trial sample for trial={self.trial_id}@{self._current_tick_id}")
            yield trial_sample

    def produce_training_sample(self, sample):
        log.debug(f"[{self.run_id}] producing a training sample for trial={self.trial_id}@{self._current_tick_id}")
        return self._produce_training_sample(self.trial_id, self._current_tick_id, sample)
