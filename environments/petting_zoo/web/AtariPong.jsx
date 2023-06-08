// Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { OBSERVER_ACTOR_CLASS, EVALUATOR_ACTOR_CLASS, PlayObserver, SimplePlay } from "@cogment/cogment-verse";

import { AtariPongPzControls } from "./AtariPongPzControls";
import { AtariPongPzFeedback } from "./AtariPongPzFeedback";

const PlayAtariPong = ({ actorParams, ...props }) => {
  const actorClassName = actorParams?.className;

  if (actorClassName === OBSERVER_ACTOR_CLASS) {
    return <PlayObserver actorParams={actorParams} {...props} />;
  }
  if (actorClassName === EVALUATOR_ACTOR_CLASS) {
    return <SimplePlay actorParams={actorParams} {...props} controls={AtariPongPzFeedback} />;
  }
  return <SimplePlay actorParams={actorParams} {...props} controls={AtariPongPzControls} />;
};

export default PlayAtariPong;
