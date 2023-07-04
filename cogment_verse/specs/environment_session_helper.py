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

from typing import Any

from .session_helper import SessionHelper


class EnvironmentSessionHelper(SessionHelper):
    """
    Cogment Verse environment session helper

    Provides additional methods to the regular Cogment environment session.

    Should be mixed in an existing actor session using `mixin`.
    """

    @classmethod
    def mixin(cls, environment_session, environment_specs):
        # Dynamically change the type of `environment_session` to inherit both it's inital type and `EnvironmentSessionHelper`
        original_cls = type(environment_session)
        environment_session.__class__ = type(original_cls.__name__ + "WithCogmentVerseHelper", (original_cls, cls), {})

        # Explicitely call `EnvironmentSessionHelper` constructor to initialze it
        cls.__init__(
            self=environment_session, environment_session=environment_session, environment_specs=environment_specs
        )

    def __init__(self, environment_session, environment_specs):
        super().__init__(
            actor_infos=environment_session.get_active_actors(),
            environment_specs=environment_specs,
            render_width=environment_session.config.render_width,
        )

    def get_action(self, tick_data: Any, actor_name: str):
        # For environments, tick_datas are events
        event = tick_data

        if not event.actions:
            return None

        actor_idx = self._get_actor_idx(actor_name)
        action_space = self._get_action_space_from_actor_idx(actor_idx)

        return action_space.deserialize(
            event.actions[actor_idx].action,
        )

    def get_player_actions(self, tick_data: Any, actor_name=None):
        event = tick_data
        if not event.actions:
            return None

        return super().get_player_actions(tick_data, actor_name)
