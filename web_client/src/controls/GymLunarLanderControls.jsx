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
import { DPad, usePressedButtons, DPAD_BUTTONS } from "./components/DPad";

const DEACTIVATED_BUTTONS_TEACHER = [];
const DEACTIVATED_BUTTONS_PLAYER = [DPAD_BUTTONS.UP];

export const GymLunarLanderEnvironments = ["gym/LunarLander-v2"];

export const GymLunarLanderControls = ({ sendAction, fps = 20, role }) => {
  const isTeacher = role === cogment_verse.HumanRole.TEACHER;
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();
  const { pressedButtons, isButtonPressed, setPressedButtons } = usePressedButtons();
  const [activeButtons, setActiveButtons] = useState([]);

  const computeAndSendAction = useCallback(
    (dt) => {
      if (pressedKeys.has("ArrowRight") || isButtonPressed(DPAD_BUTTONS.RIGHT)) {
        setActiveButtons([DPAD_BUTTONS.RIGHT]);
        sendAction(
          new cogment_verse.AgentAction({
            discreteAction: 1,
          })
        );
        return;
      } else if (pressedKeys.has("ArrowDown") || isButtonPressed(DPAD_BUTTONS.DOWN)) {
        setActiveButtons([DPAD_BUTTONS.DOWN]);
        sendAction(
          new cogment_verse.AgentAction({
            discreteAction: 2,
          })
        );
        return;
      } else if (pressedKeys.has("ArrowLeft") || isButtonPressed(DPAD_BUTTONS.LEFT)) {
        setActiveButtons([DPAD_BUTTONS.LEFT]);
        sendAction(
          new cogment_verse.AgentAction({
            discreteAction: 3,
          })
        );
        return;
      } else if (isTeacher) {
        if (pressedKeys.has("ArrowUp") || isButtonPressed(DPAD_BUTTONS.UP)) {
          setActiveButtons([DPAD_BUTTONS.UP]);
          sendAction(
            new cogment_verse.AgentAction({
              discreteAction: 0,
            })
          );
          return;
        }
        setActiveButtons([]);
        sendAction(TEACHER_NOOP_ACTION);
        return;
      }
      setActiveButtons([]);
      sendAction(
        new cogment_verse.AgentAction({
          discreteAction: 0,
        })
      );
    },
    [isButtonPressed, pressedKeys, sendAction, setActiveButtons, isTeacher]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div>
      <div>
        <DPad
          pressedButtons={pressedButtons}
          onPressedButtonsChange={setPressedButtons}
          activeButtons={activeButtons}
          disabled={paused || (isTeacher ? DEACTIVATED_BUTTONS_TEACHER : DEACTIVATED_BUTTONS_PLAYER)}
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
        {isTeacher ? <li>Up Arrow: turn off engine</li> : null}
        <li>Pause/Unpause: p</li>
      </ul>
    </div>
  );
};
