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

from .environment_specs import EnvironmentSpecs


class ActorSessionHelper:
    """
    Cogment Verse actor session helper

    Provides additional methods to the regular Cogment actor session.

    Should be mixed in an existing actor session using `mixin`.
    """

    @classmethod
    def mixin(cls, actor_session, model_registry):
        # Dynamically change the type of `actor_session` to inherit both it's inital type and `ActorSessionHelper`
        original_cls = type(actor_session)
        actor_session.__class__ = type(original_cls.__name__ + "WithCogmentVerseHelper", (original_cls, cls), {})

        # Explicitely call `ActorSessionHelper` constructor to initialze it
        cls.__init__(self=actor_session, actor_session=actor_session, model_registry=model_registry)

    def __init__(self, actor_session, model_registry):
        self.environment_specs = EnvironmentSpecs.deserialize(actor_session.config.environment_specs)
        self.action_space = self.environment_specs.get_action_space(seed=actor_session.config.seed)
        self.observation_space = self.environment_specs.get_observation_space()
        self.model_registry = model_registry

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_observation(self, event):
        if not event.observation:
            return None

        return self.observation_space.deserialize(event.observation.observation)
