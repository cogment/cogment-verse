// Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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

import { cogment_verse } from "../data_pb";

export const TEACHER_NOOP_ACTION = new cogment_verse.TeacherAction({ value: null });

export const WEB_ACTOR_NAME = "web_actor";

export const TEACHER_ACTOR_CLASS = "teacher";
export const PLAYER_ACTOR_CLASS = "player";
export const OBSERVER_ACTOR_CLASS = "observer";

export const ORCHESTRATOR_WEB_ENDPOINT =
  window.ORCHESTRATOR_WEB_ENDPOINT !== ""
    ? window.ORCHESTRATOR_WEB_ENDPOINT
    : process.env.REACT_APP_ORCHESTRATOR_WEB_ENDPOINT;
