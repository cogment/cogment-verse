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
import time
from collections import namedtuple

from cogment.session import ActorInfo, EventType, RecvAction, RecvEvent

SentReward = namedtuple("SentReward", ["value", "confidence", "to", "tick_id", "user_data"])
SentEvent = namedtuple(
    "SentEvent",
    ["tick_id", "done", "observations", "rewards", "messages", "error"],
    defaults=(0, False, [], [], [], None),
)
# Make it explicit we reexport ActorInfo
# pylint: disable=self-assigning-variable
ActorInfo = ActorInfo


class ActionData:
    def __init__(self, tick_id, timestamp):
        self.tick_id = tick_id
        self.timestamp = timestamp


class MockEnvironmentSession:
    def __init__(self, trial_id, environment_config, actor_infos, environment_impl):
        self.config = environment_config
        self._trial_id = trial_id
        self._actor_infos = actor_infos

        self._tick_id = 0
        self._done = False
        self._to_send_rewards = []
        self._to_send_messages = []
        self._sent_events_queue = asyncio.Queue()
        self._recv_events_queue = asyncio.Queue()
        self._environment_impl_error = None

        async def environment_impl_worker():
            try:
                await environment_impl(self)
            except asyncio.CancelledError as cancelled_error:
                # Raising cancellation
                raise cancelled_error
            except Exception as err:
                self._sent_events_queue.put_nowait(SentEvent(tick_id=self._tick_id, error=err))

        self._impl_task = asyncio.create_task(environment_impl_worker())

    async def terminate(self):
        self._impl_task.cancel()
        try:
            await self._impl_task
        except asyncio.CancelledError:
            pass
        self._impl_task = None

    def _produce_observations(self, observations, done):
        # Assuming there's exactly one call to `produce_observations`
        # Send what's been accumulating up until now alongside the observation
        sent_event = SentEvent(
            tick_id=self._tick_id,
            done=done,
            observations=observations,
            rewards=self._to_send_rewards,
            messages=self._to_send_messages,
        )
        self._done = done
        self._sent_events_queue.put_nowait(sent_event)
        self._tick_id += 1
        self._to_send_rewards = []
        self._to_send_messages = []

    def start(self, observations):
        self._produce_observations(observations, done=False)

    def add_reward(self, value, confidence, to, tick_id=-1, user_data=None):
        self._to_send_rewards.append(
            SentReward(value=value, confidence=confidence, to=to, tick_id=tick_id, user_data=user_data)
        )

    def produce_observations(self, observations):
        self._produce_observations(observations, done=self._done)

    def end(self, observations):
        self._produce_observations(observations, done=True)

    async def event_loop(self):
        while not self._done:
            event = await self._recv_events_queue.get()
            self._done = (
                event.type == EventType.ENDING
            )  # Will make sure the next call to produce_observations behave as `end`
            yield event

    def get_trial_id(self):
        return self._trial_id

    def get_tick_id(self):
        return self._tick_id

    def is_trial_over(self):
        return self._done

    def get_active_actors(self):
        return self._actor_infos

    async def receive_events(self):
        event = await self._sent_events_queue.get()
        if event.error:
            raise RuntimeError("Error occured while executing the environment session") from event.error
        return event

    # pylint: disable=dangerous-default-value
    def send_events(self, etype=EventType.ACTIVE, actions=[]):
        # No support for messages yet, to be added later
        event = RecvEvent(etype)

        action_data = ActionData(self._tick_id, time.time())

        event.actions = [
            RecvAction(actor_index=i, action_data=action_data, action=action) for i, action in enumerate(actions)
        ]
        self._recv_events_queue.put_nowait(event)
