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
from collections.abc import Sequence

from numpy import ndarray
from cogment.session import ActorInfo

from ..constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS, OBSERVER_ACTOR_CLASS, EVALUATOR_ACTOR_CLASS, ActorSpecType
from .action_space import Action
from .action_space import ActionSpace
from .environment_specs import EnvironmentSpecs
from .observation_space import ObservationSpace


class PlayerAction:
    """
    Wrapper around the action of a player at a given tick providing easy access to the "effective" action.

    If the player action was overriden by a teacher actor, the effective action is the override, otherwise it's the original action
    """

    def __init__(self, session_wrapper: "SessionWrapper", actor_name: str, tick_data: Any):
        self.session_wrapper = session_wrapper
        self.actor_name = actor_name
        self.tick_data = tick_data

    @property
    def original(self) -> Action:
        """
        Returns the original action of the player
        """
        return self.session_wrapper.get_action(self.tick_data, self.actor_name)

    @property
    def override(self):
        """
        If the player's action was overriden by a teacher actor, returns it, returns `None` otherwise
        """
        if len(self.session_wrapper.teacher_actors) == 0:
            return None

        num_player_actors = len(self.session_wrapper.player_actors)
        for teacher_actor_name in self.session_wrapper.teacher_actors:
            action = self.session_wrapper.get_action(self.tick_data, teacher_actor_name)

            if action.value is None:
                continue

            # For now let's take the first overriden action that matches
            if (num_player_actors == 1) or action.overridden_player == self.actor_name:
                return action

        return None

    @property
    def is_overriden(self):
        """
        Check if the action was overriden by a teacher actor
        """
        return self.override is not None

    @property
    def value(self) -> ndarray:
        """
        Value of the effective action
        """
        override = self.override
        if override is not None:
            return override.value

        return self.original.value

    @property
    def flat_value(self) -> ndarray:
        """
        Flattened value of the effective action
        """
        override = self.override
        if override is not None:
            return override.flat_value

        return self.original.flat_value


class SessionWrapper(ABC):
    """
    Cogment Verse abstract session helper, shared between the environment session and the sample producer session.
    """

    def __init__(self, actor_infos: Sequence[ActorInfo], environment_specs: EnvironmentSpecs, render_width=None):
        # Mapping actor_idx to actor_info
        self.actor_infos = actor_infos
        # Mapping actor_name to actor_idx
        self.actor_idxs = {actor_info.actor_name: actor_idx for (actor_idx, actor_info) in enumerate(self.actor_infos)}
        self
        self.actors = [actor_info.actor_name for actor_info in self.actor_infos]
        self.actors_spec_type = {actor_info.actor_name: ActorSpecType.from_config(actor_info.actor_class_name) for actor_info in self.actor_infos}

        self.environment_specs = environment_specs
        self.render_width = render_width

    def get_observation_space(self, actor_name: str) -> ObservationSpace:
        return self.environment_specs[self.actors_spec_type[actor_name]].get_observation_space(self.render_width)

    def _get_actor_idx(self, actor_name: str) -> int:
        actor_idx = self.actor_idxs.get(actor_name)

        if actor_idx is None:
            raise RuntimeError(f"No actor with name [{actor_name}] found!")

        return actor_idx

    def _get_actor_name(self, actor_idx: int) -> str:
        return self.actor_infos[actor_idx].actor_name

    def get_actor_class_name(self, actor_name: str) -> str:
        actor_idx = self._get_actor_idx(actor_name)
        return self.actor_infos[actor_idx].actor_class_name

    def _get_action_space_from_actor_idx(self, actor_idx: int) -> ActionSpace:
        actor_name  = self._get_actor_name(actor_idx)
        actor_specs = self.actors_spec_type[actor_name]
        return self.environment_specs[actor_specs].get_action_space(self.get_actor_class_name(actor_name))

    def get_action_space(self, actor_name: str) -> ActionSpace:
        actor_specs = self.actors_spec_type[actor_name]
        return self.environment_specs[actor_specs].get_action_space(self.get_actor_class_name(actor_name))

    def _list_actors_by_class(self, actor_class_name: str) -> Sequence[str]:
        return [
            actor_info.actor_name for actor_info in self.actor_infos if actor_info.actor_class_name == actor_class_name
        ]

    @property
    def player_actors(self) -> Sequence[str]:
        """
        List player actors in the current trial
        """
        return self._list_actors_by_class(PLAYER_ACTOR_CLASS)

    @property
    def teacher_actors(self) -> Sequence[str]:
        """
        List teacher actors in the current trial
        """
        return self._list_actors_by_class(TEACHER_ACTOR_CLASS)

    @property
    def observer_actors(self) -> Sequence[str]:
        """
        List observer actors in the current trial
        """
        return self._list_actors_by_class(OBSERVER_ACTOR_CLASS)

    @property
    def evaluator_actors(self) -> Sequence[str]:
        """
        List evaluator actors in the current trial
        """
        return self._list_actors_by_class(EVALUATOR_ACTOR_CLASS)

    @abstractmethod
    def get_action(self, tick_data: Any, actor_name: str) -> Action:
        """
        Return the cogment verse action of a given actor at a tick.

        If no action, returns None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, tick_data: Any, actor_name: str):
        """
        Return the cogment verse observation of a given actor at a tick.

        If no observation, returns None.
        """
        raise NotImplementedError

    def get_player_actions(self, tick_data: Any, actor_name: str = None) -> PlayerAction:
        """
        Return the cogment verse player action of a given actor at a tick.

        If only a single player actor is present, no `actor_name` is required.

        If no action, returns None.
        """
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

    def get_player_observations(self, tick_data: Any, actor_name: str = None):
        if actor_name is None:
            observations = [self.get_observation(tick_data, actor_name) for player_actor_name in self.player_actors]
            if len(observations) == 0:
                raise RuntimeError("No player actors")
            if len(observations) > 1:
                raise RuntimeError("More than 1 player actor, please provide an actor name")
            return observations[0]

        observations = [
            self.get_observation(tick_data, actor_name)
            for player_actor_name in self.player_actors
            if player_actor_name == actor_name
        ]
        if len(observations) == 0:
            raise RuntimeError(f"No player actors having name [{actor_name}]")
        return observations[0]
