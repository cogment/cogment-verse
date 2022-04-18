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

export const GymLunarLanderContinuousEnvironments = ["gym/LunarLanderContinuous-v2"];
export const GymLunarLanderContinuousControls = ({ sendAction, fps = 30, role }) => {
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();

  const [lastAction, setLastAction] = useState(null);
  const computeAndSendAction = useCallback(
    (dt) => {
      if (pressedKeys.size === 0 && role === cogment_verse.HumanRole.TEACHER) {
        sendAction(TEACHER_NOOP_ACTION);
        return;
      }

      const action_params = {
        continuousAction: {
          data: [0, 0],
        },
      };

      const fast = pressedKeys.has("Shift");

      if (pressedKeys.has("ArrowLeft")) {
        action_params.continuousAction.data[1] = fast ? -1.0 : -0.51;
      } else if (pressedKeys.has("ArrowRight")) {
        action_params.continuousAction.data[1] = fast ? 1.0 : 0.51;
      }

      if (pressedKeys.has("ArrowDown")) {
        action_params.continuousAction.data[0] = fast ? 1 : 0.01;
      }

      const action = new cogment_verse.AgentAction(action_params);
      sendAction(action);
      setLastAction(action);
    },
    [pressedKeys, sendAction, setLastAction, role]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div>
      <button onClick={togglePause}>{paused ? "Resume" : "Pause"}</button>
      <div>{currentFps.toFixed(2)} fps</div>
      <div>{lastAction != null ? JSON.stringify(lastAction.continuousAction.data) : "N/A"}</div>
      <ul>
        <li>Left Arrow: fire left engine</li>
        <li>Right Arrow: fire right engine</li>
        <li>Down Arrow: fire main engine</li>
        <li>Shift: maintain pressed for 100% power</li>
        <li>Pause/Unpause: p</li>
      </ul>
    </div>
  );
};
