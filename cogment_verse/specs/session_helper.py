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

from abc import ABC, abstractmethod
from typing import Any

from ..constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS, OBSERVER_ACTOR_CLASS, EVALUATOR_ACTOR_CLASS


class PlayerAction:
    def __init__(self, session_helper, actor_name, tick_data):
        self.session_helper = session_helper
        self.actor_name = actor_name
        self.tick_data = tick_data

    @property
    def original(self):
        return self.session_helper.get_action(self.tick_data, self.actor_name)

    @property
    def override(self):
        if len(self.session_helper.teacher_actors) == 0:
            return None

        num_player_actors = len(self.session_helper.player_actors)
        for teacher_actor_name in self.session_helper.teacher_actors:
            action = self.session_helper.get_action(self.tick_data, teacher_actor_name)

            if action.value is None:
                continue

            # For now let's take the first overriden action that matches
            if (num_player_actors == 1) or action.overridden_player == self.actor_name:
                return action

        return None

    @property
    def is_overriden(self):
        return self.override is not None

    @property
    def value(self):
        override = self.override
        if override is not None:
            return override.value

        return self.original.value

    @property
    def flat_value(self):
        override = self.override
        if override is not None:
            return override.flat_value

        return self.original.flat_value


class SessionHelper(ABC):
    def __init__(self, actor_infos, environment_specs, render_width=None):
        # Mapping actor_idx to actor_info
        self.actor_infos = actor_infos
        # Mapping actor_name to actor_idx
        self.actor_idxs = {actor_info.actor_name: actor_idx for (actor_idx, actor_info) in enumerate(self.actor_infos)}

        self.environment_specs = environment_specs
        self.observation_space = environment_specs.get_observation_space(render_width)

    def get_observation_space(self, _actor_name):
        # TODO take the _actor_name into account
        return self.observation_space

    def _get_actor_idx(self, actor_name: str):
        actor_idx = self.actor_idxs.get(actor_name)

        if actor_idx is None:
            raise RuntimeError(f"No actor with name [{actor_name}] found!")

        return actor_idx

    def get_actor_class_name(self, actor_name: str):
        actor_idx = self._get_actor_idx(actor_name)
        return self.actor_infos[actor_idx].actor_class_name

    def _get_action_space_from_actor_idx(self, actor_idx: int):
        return self.environment_specs.get_action_space(self.actor_infos[actor_idx].actor_class_name)

    def get_action_space(self, actor_name: str):
        return self.environment_specs.get_action_space(self.get_actor_class_name(actor_name))

    def _list_actors_by_class(self, actor_class_name):
        return [
            actor_info.actor_name for actor_info in self.actor_infos if actor_info.actor_class_name == actor_class_name
        ]

    @property
    def player_actors(self):
        return self._list_actors_by_class(PLAYER_ACTOR_CLASS)

    @property
    def teacher_actors(self):
        return self._list_actors_by_class(TEACHER_ACTOR_CLASS)

    @property
    def observer_actors(self):
        return self._list_actors_by_class(OBSERVER_ACTOR_CLASS)

    @property
    def evaluator_actors(self):
        return self._list_actors_by_class(EVALUATOR_ACTOR_CLASS)

    @abstractmethod
    def get_action(self, tick_data: Any, actor_name: str):
        pass

    def get_player_action(self, tick_data: Any, actor_name: str = None):
        if actor_name is None:
            actions = [PlayerAction(self, actor_name, tick_data) for actor_name in self.player_actors]
            if len(actions) == 0:
                raise RuntimeError("No player actors")
            if len(actions) > 1:
                raise RuntimeError("More than 1 player actor, please provide an actor name")
            return actions[0]

        actions = [
            PlayerAction(self, actor_name, tick_data)
            for player_actor_name in self.player_actors
            if player_actor_name == actor_name
        ]
        if len(actions) == 0:
            raise RuntimeError(f"No player actors having name [{actor_name}]")
        return actions[0]
