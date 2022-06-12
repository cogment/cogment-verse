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
import numpy as np

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

        rng = np.random.default_rng(config.seed if config.seed is not None else 0)

        async for event in actor_session.all_events():
            if event.observation and event.type == cogment.EventType.ACTIVE:
                if (
                    event.observation.observation.HasField("current_player")
                    and event.observation.observation.current_player != actor_session.name
                ):
                    # Not the turn of the agent
                    actor_session.do_action(PlayerAction())
                    continue
                [action_value] = sample_space(action_space, rng=rng, mask=event.observation.observation.action_mask)
                actor_session.do_action(PlayerAction(value=action_value))
