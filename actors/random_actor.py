# Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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
from cogment_verse.constants import ActorSpecType

from cogment_verse.specs import PLAYER_ACTOR_CLASS, EnvironmentSpecs
from environments.petting_zoo.mpe_environment import MpeSpecType


class RandomActor:
    def __init__(self, _cfg):
        pass

    def get_actor_classes(self):
        return [PLAYER_ACTOR_CLASS] + MpeSpecType.values

    async def impl(self, actor_session):
        actor_session.start()
        async for event in actor_session.all_events():
            observation = actor_session.get_observation(event)
            if observation and event.type == cogment.EventType.ACTIVE:
                if observation.current_player is not None and observation.current_player.name != actor_session.name:
                    # Not the turn of the agent
                    action = actor_session.get_action_space().create()
                    actor_session.do_action(actor_session.get_action_space().serialize(action))
                    continue

                action = actor_session.get_action_space().sample(mask=observation.action_mask)
                actor_session.do_action(actor_session.get_action_space().serialize(action))
