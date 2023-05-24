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

import { useCallback, useState } from "react";
import { useDocumentKeypressListener, usePressedKeys } from "../hooks/usePressedKeys";
import { useRealTimeUpdate } from "../hooks/useRealTimeUpdate";
import { DPad, usePressedButtons, DPAD_BUTTONS } from "../components/DPad";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";
import { serializePlayerAction, Space } from "../utils/spaceSerialization";

const ACTION_SPACE = new Space({
    discrete: {
      n: 6,
    },
  });


export const OvercookedRealTimeEnvironments = ["environments.overcooked_adapter.OvercookedEnvironment/overcooked-real-time"];
export const OvercookedRealTimeControls = ({ sendAction, fps = 10, actorClass, ...props }) => {
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();
  const { pressedButtons, isButtonPressed, setPressedButtons } = usePressedButtons();
  const [activeButtons, setActiveButtons] = useState([]);

  const computeAndSendAction = useCallback(
    (dt) => {
      if (pressedKeys.has("ArrowLeft") || isButtonPressed(DPAD_BUTTONS.LEFT)) {
        setActiveButtons([DPAD_BUTTONS.LEFT]);
        sendAction(serializePlayerAction(ACTION_SPACE, 3));
        return;
      } else if (pressedKeys.has("ArrowRight") || isButtonPressed(DPAD_BUTTONS.RIGHT)) {
        setActiveButtons([DPAD_BUTTONS.RIGHT]);
        sendAction(serializePlayerAction(ACTION_SPACE, 2));
        return;
      } else if (pressedKeys.has("ArrowUp") || isButtonPressed(DPAD_BUTTONS.UP)) {
        setActiveButtons([DPAD_BUTTONS.UP]);
        sendAction(serializePlayerAction(ACTION_SPACE, 0));
        return;
      } else if (pressedKeys.has("ArrowDown") || isButtonPressed(DPAD_BUTTONS.DOWN)) {
        setActiveButtons([DPAD_BUTTONS.DOWN]);
        sendAction(serializePlayerAction(ACTION_SPACE, 1));
        return;
      } else if (pressedKeys.has("Enter")) {
        sendAction(serializePlayerAction(ACTION_SPACE, 5));
        return;
      } else if (pressedKeys.has("Shift") ) {
        sendAction(serializePlayerAction(ACTION_SPACE, 4));
        return;
      }
    },
    [isButtonPressed, pressedKeys, sendAction]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div {...props}>
      <div className="flex flex-row p-5 justify-center">
        <DPad
          pressedButtons={pressedButtons}
          onPressedButtonsChange={setPressedButtons}
          activeButtons={activeButtons}
        />
      </div>
      <div className="flex flex-row gap-1">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
        <FpsCounter className="flex-none w-fit" value={currentFps} />
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right/Up/Down Arrows", "Move end-effector left/right/up/down"],
          ["Enter", "Interact"],
          ["Space", "Skip time step"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
