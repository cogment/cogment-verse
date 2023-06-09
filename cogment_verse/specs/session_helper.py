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

from ..constants import PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS, OBSERVER_ACTOR_CLASS, EVALUATOR_ACTOR_CLASS
from .environment_specs import EnvironmentSpecs


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


class _ActorInfo:
    def __init__(self, name, class_name):
        self.name = name
        self.class_name = class_name


class SessionHelper(ABC):
    def __init__(self, actor_infos, environment_specs, render_width=None):
        # Mapping actor_idx to actor_info
        self.actor_infos = actor_infos
        # Mapping actor_name to actor_idx
        self.actor_idxs = {actor_info.name: actor_idx for (actor_idx, actor_info) in enumerate(self.actor_infos)}

        self.environment_specs = environment_specs
        self.observation_space = environment_specs.get_observation_space(render_width)

    def get_observation_space(self, _actor_idx_or_name):
        return self.observation_space

    def get_actor_idx(self, actor_idx_or_name):
        if isinstance(actor_idx_or_name, int):
            return actor_idx_or_name

        actor_idx = self.actor_idxs.get(actor_idx_or_name)

        if actor_idx is None:
            raise RuntimeError(f"No actor with name {actor_idx_or_name} found!")

        return actor_idx

    def get_actor_name(self, actor_idx_or_name):
        if isinstance(actor_idx_or_name, int):
            return self.actor_infos[actor_idx_or_name].name

        return actor_idx_or_name

    def get_actor_class_name(self, actor_idx_or_name):
        actor_idx = self.get_actor_idx(actor_idx_or_name)
        return self.actor_infos[actor_idx].class_name

    def get_action_space(self, actor_idx_or_name):
        return self.environment_specs.get_action_space(self.get_actor_class_name(actor_idx_or_name))

    def _list_actors_by_class(self, actor_class_name):
        return [actor_info.name for actor_info in self.actor_infos if actor_info.class_name == actor_class_name]

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
    def get_action(self, tick_data, actor_idx_or_name):
        pass

    def get_player_actions(self, tick_data):
        return [PlayerAction(self, actor_name, tick_data) for actor_name in self.player_actors]


class ActorSessionHelper:
    def __init__(self, actor_session):
        self.config = actor_session.config
        self.environment_specs = EnvironmentSpecs.deserialize(self.config.environment_specs)
        self.action_space = self.environment_specs.get_action_space(seed=self.config.seed)
        self.observation_space = self.environment_specs.get_observation_space()

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_observation(self, event):
        if not event.observation:
            return None

        return self.observation_space.deserialize(event.observation.observation)


class EnvironmentSessionHelper(SessionHelper):
    def __init__(self, environment_session, environment_specs):
        super().__init__(
            actor_infos=[
                _ActorInfo(name=actor_info.actor_name, class_name=actor_info.actor_class_name)
                for actor_info in environment_session.get_active_actors()
            ],
            environment_specs=environment_specs,
            render_width=environment_session.config.render_width,
        )

    def get_action(self, tick_data, actor_idx_or_name):
        # For environments, tick_datas are events
        event = tick_data

        if not event.actions:
            return None

        actor_idx = self.get_actor_idx(actor_idx_or_name)
        action_space = self.get_action_space(actor_idx)

        return action_space.deserialize(
            event.actions[actor_idx].action,
        )

    def get_player_actions(self, tick_data):
        event = tick_data
        if not event.actions:
            return []

        return super().get_player_actions(tick_data)


class SampleProducerSessionHelper(SessionHelper):
    def __init__(self, sample_producer_session):
        actor_infos = []
        environment_specs = None
        for actor_params in sample_producer_session.trial_info.parameters.actors:
            actor_infos.append(_ActorInfo(name=actor_params.name, class_name=actor_params.class_name))
            if environment_specs is None:
                environment_specs = EnvironmentSpecs.deserialize(actor_params.config.environment_specs)
        super().__init__(
            actor_infos=actor_infos,
            environment_specs=environment_specs,
        )

    def get_observation(self, tick_data, actor_idx_or_name):
        # For sample producers, tick_datas are samples
        sample = tick_data

        actor_name = self.get_actor_name(actor_idx_or_name)
        observation_space = self.get_observation_space(actor_name)

        return observation_space.deserialize(sample.actors_data[actor_name].observation)

    def get_player_observations(self, tick_data):
        return [self.get_observation(tick_data, actor_name) for actor_name in self.player_actors]

    def get_action(self, tick_data, actor_idx_or_name):
        # For sample producers, tick_datas are samples
        sample = tick_data

        actor_name = self.get_actor_name(actor_idx_or_name)
        action_space = self.get_action_space(actor_name)
        return action_space.deserialize(sample.actors_data[actor_name].action)
