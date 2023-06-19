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

from cogment.environment import EnvironmentSession
from cogment.session import RecvEvent

from .session_wrapper import SessionWrapper, PlayerAction
from .environment_specs import EnvironmentSpecs
from .action_space import Action


class EnvironmentSessionWrapper(SessionWrapper):
    """
    Cogment Verse environment session helper

    Provides additional methods to the regular Cogment environment session.

    Should be mixed in an existing environment session using `mixin`.
    """

    @classmethod
    def mixin(cls, environment_session: EnvironmentSession, environment_specs: EnvironmentSpecs):
        assert isinstance(environment_session, EnvironmentSession)

        # Dynamically change the type of `environment_session` to inherit both it's inital type and `EnvironmentSessionWrapper`
        original_cls = type(environment_session)
        environment_session.__class__ = type("Wrapped" + original_cls.__name__, (original_cls, cls), {})

        # Explicitely call `SessionWrapper` constructor to initialze it
        SessionWrapper.__init__(
            self=environment_session,
            actor_infos=environment_session.get_active_actors(),
            environment_specs=environment_specs,
            render_width=environment_session.config.render_width,
        )

    def __init__(self):
        """
        Do not initialize directly, use `EnvironmentSessionWrapper.mixin` instead
        """

        super().__init__(actor_infos=[], environment_specs=None)
        raise NotImplementedError("`EnvironmentSessionWrapper` should not be initialized directly")

    def get_action(self, tick_data: Any, actor_name: str) -> Action:
        # For environments, tick_datas are events
        event: RecvEvent = tick_data

        if not event.actions:
            return None

        actor_idx = self._get_actor_idx(actor_name)
        action_space = self._get_action_space_from_actor_idx(actor_idx)

        return action_space.deserialize(
            event.actions[actor_idx].action,
        )

    def get_player_actions(self, tick_data: Any, actor_name: str = None) -> PlayerAction:
        event = tick_data
        if not event.actions:
            return None

        return super().get_player_actions(tick_data, actor_name)
