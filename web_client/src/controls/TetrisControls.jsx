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
import { createLookup } from "./utils/controlLookup";
import { TEACHER_NOOP_ACTION } from "./utils/constants";

const TETRIS_LOOKUP = createLookup();
TETRIS_LOOKUP.setAction([], new cogment_verse.AgentAction({ discreteAction: 0 }));
TETRIS_LOOKUP.setAction(["TURN_CW"], new cogment_verse.AgentAction({ discreteAction: 1 }));
TETRIS_LOOKUP.setAction(["TURN_CCW"], new cogment_verse.AgentAction({ discreteAction: 2 }));
TETRIS_LOOKUP.setAction(["RIGHT"], new cogment_verse.AgentAction({ discreteAction: 3 }));
TETRIS_LOOKUP.setAction(["RIGHT", "TURN_CW"], new cogment_verse.AgentAction({ discreteAction: 4 }));
TETRIS_LOOKUP.setAction(["RIGHT", "TURN_CCW"], new cogment_verse.AgentAction({ discreteAction: 5 }));
TETRIS_LOOKUP.setAction(["LEFT"], new cogment_verse.AgentAction({ discreteAction: 6 }));
TETRIS_LOOKUP.setAction(["LEFT", "TURN_CW"], new cogment_verse.AgentAction({ discreteAction: 7 }));
TETRIS_LOOKUP.setAction(["LEFT", "TURN_CCW"], new cogment_verse.AgentAction({ discreteAction: 8 }));
TETRIS_LOOKUP.setAction(["DOWN"], new cogment_verse.AgentAction({ discreteAction: 9 }));
TETRIS_LOOKUP.setAction(["DOWN", "TURN_CW"], new cogment_verse.AgentAction({ discreteAction: 10 }));
TETRIS_LOOKUP.setAction(["DOWN", "TURN_CCW"], new cogment_verse.AgentAction({ discreteAction: 11 }));

export const TetrisEnvironments = ["tetris/TetrisA-v0"];
export const TetrisControls = ({ sendAction, fps = 30, role }) => {
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

      const controls = [];
      if (pressedKeys.has("ArrowLeft")) {
        controls.push("LEFT");
      } else if (pressedKeys.has("ArrowRight")) {
        controls.push("RIGHT");
      } else if (pressedKeys.has("ArrowDown")) {
        controls.push("DOWN");
      }
      if (pressedKeys.has("w")) {
        controls.push("TURN_CCW");
      } else if (pressedKeys.has("x")) {
        controls.push("TURN_CW");
      }
      const action = TETRIS_LOOKUP.getAction(controls);
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
        <li>Left/Right Arrows: Move piece left/right</li>
        <li>Down Arrow: Send piece down</li>
        <li>w/x: Turn piece couterclockwise/clockwise</li>
        <li>Pause/Unpause: p</li>
      </ul>
    </div>
  );
};
