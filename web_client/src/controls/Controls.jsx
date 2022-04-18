// Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
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
import * as data_pb from "../data_pb";
import { ObserverControls } from "./ObserverControls";
import { GymLunarLanderControls, GymLunarLanderEnvironments } from "./GymLunarLanderControls";
import {
  GymLunarLanderContinuousControls,
  GymLunarLanderContinuousEnvironments,
} from "./GymLunarLanderContinuousControls";
import { GymCartPoleEnvironments, GymCartPoleControls } from "./GymCartPoleControls";
import { GymMountainCarEnvironments, GymMountainCarControls } from "./GymMountainCarControls";
import { AtariPitfallEnvironments, AtariPitfallControls } from "./AtariPitfallControls";
import { TetrisEnvironments, TetrisControls } from "./TetrisControls";

const CONTROLS = [
  { environments: GymLunarLanderEnvironments, component: GymLunarLanderControls },
  { environments: GymLunarLanderContinuousEnvironments, component: GymLunarLanderContinuousControls },
  { environments: GymCartPoleEnvironments, component: GymCartPoleControls },
  { environments: GymMountainCarEnvironments, component: GymMountainCarControls },
  { environments: AtariPitfallEnvironments, component: AtariPitfallControls },
  { environments: TetrisEnvironments, component: TetrisControls },
];

export const Controls = ({ environment, role, sendAction, fps = 30 }) => {
  const ControlsComponent = useMemo(() => {
    if (data_pb.cogment_verse.HumanRole.OBSERVER === role) {
      return ObserverControls;
    }
    if ([data_pb.cogment_verse.HumanRole.PLAYER, data_pb.cogment_verse.HumanRole.TEACHER].includes(role)) {
      const control = CONTROLS.find(({ environments }) => environments.includes(environment));
      if (control == null) {
        return () => <div>{environment} is not playable</div>;
      }
      return control.component;
    }
    return () => <div>Unknown role "{role}"</div>;
  }, [environment, role]);

  return <ControlsComponent sendAction={sendAction} fps={fps} role={role} environment={environment} />;
};
