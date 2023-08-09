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

from cogment.actor import ActorSession
from cogment.model_registry_v2 import ModelRegistry
from cogment_verse.constants import ActorSpecType
from cogment_verse.specs.actor_specs import ActorSpecs

from .environment_specs import EnvironmentSpecs
from .action_space import ActionSpace
from .observation_space import ObservationSpace, Observation


class ActorSessionWrapper:
    """
    Cogment Verse actor session helper

    Provides additional methods to the regular Cogment actor session.

    Should be mixed in an existing actor session using `mixin`.
    """

    @classmethod
    def mixin(cls, actor_session: ActorSession, model_registry: ModelRegistry):
        assert isinstance(actor_session, ActorSession)

        # Dynamically change the type of `actor_session` to inherit both it's inital type and `ActorSessionWrapper`
        original_cls = type(actor_session)
        actor_session.__class__ = type("Wrapped" + original_cls.__name__, (original_cls, cls), {})

        # Explicitely call `ActorSessionWrapper` constructor to initialze it
        # TODO: load spec_type from an Enum instead of class name string
        actor_session.spec_type = actor_session.config.spec_type
        actor_session.environment_specs = EnvironmentSpecs.deserialize(actor_session.config.environment_specs)
        actor_session.actor_specs = actor_session.environment_specs[actor_session.spec_type]
        actor_session.action_space = actor_session.actor_specs.get_action_space(seed=actor_session.config.seed)
        actor_session.observation_space = actor_session.actor_specs.get_observation_space()
        actor_session.model_registry = model_registry

    def __init__(self):
        """
        Do not initialize directly, use `ActorSessionWrapper.mixin` instead
        """
        super().__init__()
        self.spec_type: ActorSpecType = None
        self.environment_specs: EnvironmentSpecs = None
        self.actor_specs: ActorSpecs = None
        self.action_space: ActionSpace = None
        self.observation_space: ObservationSpace = None
        self.model_registry: ModelRegistry = None
        raise NotImplementedError("`ActorSessionWrapper` should not be initialized directly")

    def get_action_space(self) -> ActionSpace:
        return self.action_space

    def get_observation_space(self) -> ObservationSpace:
        return self.observation_space

    def get_observation(self, event) -> Observation:
        """
        Return the cogment verse observation for the current event.

        If the event does not contain an observation, return None.
        """
        if not event.observation:
            return None

        return self.observation_space.deserialize(event.observation.observation)
