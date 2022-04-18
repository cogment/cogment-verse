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

import { useCallback, useState } from "react";
import { cogment_verse } from "../data_pb";
import { useDocumentKeypressListener, usePressedKeys } from "./hooks/usePressedKeys";
import { useRealTimeUpdate } from "./hooks/useRealTimeUpdate";
import { TEACHER_NOOP_ACTION } from "./utils/constants";

export const GymLunarLanderEnvironments = ["gym/LunarLander-v2"];
export const GymLunarLanderControls = ({ sendAction, fps = 30, role }) => {
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();

  const computeAndSendAction = useCallback(
    (dt) => {
      if (pressedKeys.size === 0 && role === cogment_verse.HumanRole.TEACHER) {
        sendAction(TEACHER_NOOP_ACTION);
        return;
      }
      const action = new cogment_verse.AgentAction({
        discreteAction: 0,
      });

      if (pressedKeys.has("ArrowLeft")) {
        action.discreteAction = 1;
      } else if (pressedKeys.has("ArrowDown")) {
        action.discreteAction = 2;
      } else if (pressedKeys.has("ArrowRight")) {
        action.discreteAction = 3;
      }

      sendAction(action);
    },
    [pressedKeys, sendAction, role]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div>
      <button onClick={togglePause}>{paused ? "Resume" : "Pause"}</button>
      <div>{currentFps.toFixed(2)} fps</div>
      <ul>
        <li>Left Arrow: fire left engine</li>
        <li>Right Arrow: fire right engine</li>
        <li>Down Arrow: fire main engine</li>
        <li>Pause/Unpause: p</li>
      </ul>
    </div>
  );
};
