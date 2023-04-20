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
import { RealTimeObserverControls } from "./RealTimeObserverControls";
import { TurnBasedObserverControls } from "./TurnBasedObserverControls";
import { GymLunarLanderControls, GymLunarLanderEnvironments } from "./GymLunarLanderControls";
import {
  GymLunarLanderContinuousControls,
  GymLunarLanderContinuousEnvironments,
} from "./GymLunarLanderContinuousControls";
import { ConnectFourControls, ConnectFourEnvironments } from "./ConnectFourControls";
import { GymCartPoleEnvironments, GymCartPoleControls } from "./GymCartPoleControls";
import { GymMountainCarEnvironments, GymMountainCarControls } from "./GymMountainCarControls";
import { AtariPitfallEnvironments, AtariPitfallControls } from "./AtariPitfallControls";
import { TetrisEnvironments, TetrisControls } from "./TetrisControls";
import { AtariPongPzEnvironments, AtariPongPzControls } from "./AtariPongPzControls";
import { AtariPongPzHfbEnvironments, AtariPongPzFeedback } from "./AtariPongPzFeedback";
import { actorClassEnum } from "../utils/constants";

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
];

export const Controls = ({ environment, actorClass, sendAction, fps, turnBased, observation, tickId }) => {
  const ControlsComponent = useMemo(() => {
    if (actorClassEnum.OBSERVER === actorClass) {
      if (turnBased) {
        return TurnBasedObserverControls;
      }
      return RealTimeObserverControls;
    }
    if ([actorClassEnum.PLAYER, actorClassEnum.TEACHER, actorClassEnum.EVALUATOR].includes(actorClass)) {
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
