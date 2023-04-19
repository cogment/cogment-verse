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

import { useMemo } from "react";
import { EVALUATOR_ACTOR_CLASS, OBSERVER_ACTOR_CLASS, PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS } from "../utils/constants";
import { AtariPitfallControls, AtariPitfallEnvironments } from "./AtariPitfallControls";
import { AtariPongPzControls, AtariPongPzEnvironments } from "./AtariPongPzControls";
import { AtariPongPzFeedback, AtariPongPzHfbEnvironments } from "./AtariPongPzFeedback";
import { ConnectFourControls, ConnectFourEnvironments } from "./ConnectFourControls";
import { GymCartPoleControls, GymCartPoleEnvironments } from "./GymCartPoleControls";
import {
  GymLunarLanderContinuousControls,
  GymLunarLanderContinuousEnvironments,
} from "./GymLunarLanderContinuousControls";
import { GymLunarLanderControls, GymLunarLanderEnvironments } from "./GymLunarLanderControls";
import { GymMountainCarControls, GymMountainCarEnvironments } from "./GymMountainCarControls";
import { OvercookedControls, OvercookedEnvironments } from "./OvercookedControls";
import { OvercookedTurnBasedControls, OvercookedTurnBasedEnvironments } from "./OvercookedTurnBasedControls";
import { RealTimeObserverControls } from "./RealTimeObserverControls";
import { TetrisControls, TetrisEnvironments } from "./TetrisControls";
import { TurnBasedObserverControls } from "./TurnBasedObserverControls";

const CONTROLS = [
  { environments: GymLunarLanderEnvironments, component: GymLunarLanderControls },
  { environments: GymLunarLanderContinuousEnvironments, component: GymLunarLanderContinuousControls },
  { environments: GymCartPoleEnvironments, component: GymCartPoleControls },
  { environments: GymMountainCarEnvironments, component: GymMountainCarControls },
  { environments: AtariPitfallEnvironments, component: AtariPitfallControls },
  { environments: TetrisEnvironments, component: TetrisControls },
  { environments: ConnectFourEnvironments, component: ConnectFourControls },
  { environments: AtariPongPzEnvironments, component: AtariPongPzControls },
  { environments: AtariPongPzHfbEnvironments, component: AtariPongPzFeedback },
  { environments: OvercookedEnvironments, component: OvercookedControls },
  { environments: OvercookedTurnBasedEnvironments, component: OvercookedTurnBasedControls },
];

export const Controls = ({ environment, actorClass, sendAction, fps, turnBased, observation, tickId }) => {
  const ControlsComponent = useMemo(() => {
    if (OBSERVER_ACTOR_CLASS === actorClass) {
      if (turnBased) {
        return TurnBasedObserverControls;
      }
      return RealTimeObserverControls;
    }
    if ([PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS, EVALUATOR_ACTOR_CLASS].includes(actorClass)) {
      const control = CONTROLS.find(({ environments }) => environments.includes(environment));
      if (control == null) {
        return () => <div>{environment} is not playable</div>;
      }
      return control.component;
    }
    return () => <div>Unknown actor class "{actorClass}"</div>;
  }, [environment, actorClass, turnBased]);

  return (
    <ControlsComponent
      sendAction={sendAction}
      fps={fps}
      actorClass={actorClass}
      environment={environment}
      observation={observation}
      tickId={tickId}
    />
  );
};
