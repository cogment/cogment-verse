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

import { useJoystickState, Joystick } from "./components/Joystick";

export const GymLunarLanderContinuousEnvironments = ["gym/LunarLanderContinuous-v2"];
export const GymLunarLanderContinuousControls = ({ sendAction, fps = 20, role }) => {
  const isTeacher = role === cogment_verse.HumanRole.TEACHER;
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();
  const { joystickPosition, isJoystickActive, setJoystickState } = useJoystickState();

  const [isKeyboardActive, setKeyboardActive] = useState(false);

  const computeAndSendAction = useCallback(
    (dt) => {
      if (isJoystickActive) {
        sendAction(
          new cogment_verse.AgentAction({
            continuousAction: {
              data: [joystickPosition[1], -joystickPosition[0]],
            },
          })
        );
        setKeyboardActive(false);
        return;
      }

      let keyboardAction = null;
      const fast = pressedKeys.has("Shift");
      if (pressedKeys.has("ArrowLeft")) {
        keyboardAction = [0, 0];
        keyboardAction[1] = fast ? 1.0 : 0.75;
      } else if (pressedKeys.has("ArrowRight")) {
        keyboardAction = [0, 0];
        keyboardAction[1] = fast ? -1.0 : -0.75;
      }

      if (pressedKeys.has("ArrowDown")) {
        keyboardAction = keyboardAction || [0, 0];
        keyboardAction[0] = fast ? 1 : 0.25;
      }

      if (keyboardAction != null) {
        sendAction(
          new cogment_verse.AgentAction({
            continuousAction: {
              data: keyboardAction,
            },
          })
        );
        setKeyboardActive(true);
        setJoystickState([-keyboardAction[1], keyboardAction[0]], false);
        return;
      }

      if (isTeacher) {
        sendAction(TEACHER_NOOP_ACTION);
      } else {
        sendAction(
          new cogment_verse.AgentAction({
            continuousAction: {
              data: [0, 0],
            },
          })
        );
      }
      setKeyboardActive(false);
      setJoystickState([0, 0], false);
    },
    [isTeacher, pressedKeys, joystickPosition, isJoystickActive, setJoystickState, setKeyboardActive, sendAction]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div>
      <div>
        <Joystick
          position={joystickPosition}
          active={isJoystickActive}
          onChange={setJoystickState}
          disabled={paused || isKeyboardActive}
        />
      </div>
      <div>
        <button onClick={togglePause}>{paused ? "Resume" : "Pause"}</button>
      </div>
      <div>{currentFps.toFixed(2)} fps</div>
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
