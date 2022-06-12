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

import cogment

from cogment_verse.specs import (
    PLAYER_ACTOR_CLASS,
    PlayerAction,
    sample_space,
)


class RandomActor:
    def __init__(self, _cfg):
        pass

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS]

    async def impl(self, actor_session):
        actor_session.start()

        config = actor_session.config

        action_space = config.environment_specs.action_space

        # TODO this is something that could be configured
        random_seed = 0

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                [action_value] = sample_space(action_space, seed=random_seed + actor_session.get_tick_id())
                actor_session.do_action(PlayerAction(value=action_value))
