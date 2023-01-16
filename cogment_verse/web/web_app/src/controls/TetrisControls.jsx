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

import { useCallback, useState } from "react";
import { useDocumentKeypressListener, usePressedKeys } from "../hooks/usePressedKeys";
import { useRealTimeUpdate } from "../hooks/useRealTimeUpdate";
import { createLookup } from "../utils/controlLookup";
import { TEACHER_ACTOR_CLASS } from "../utils/constants";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";
import { TEACHER_NOOP_ACTION, serializePlayerAction, Space } from "../utils/spaceSerialization";

const ACTION_SPACE = new Space({
  discrete: {
    n: 12,
  },
});

const TETRIS_LOOKUP = createLookup();
TETRIS_LOOKUP.setAction([], serializePlayerAction(ACTION_SPACE, 0));
TETRIS_LOOKUP.setAction(["TURN_CW"], serializePlayerAction(ACTION_SPACE, 1));
TETRIS_LOOKUP.setAction(["TURN_CCW"], serializePlayerAction(ACTION_SPACE, 2));
TETRIS_LOOKUP.setAction(["RIGHT"], serializePlayerAction(ACTION_SPACE, 3));
TETRIS_LOOKUP.setAction(["RIGHT", "TURN_CW"], serializePlayerAction(ACTION_SPACE, 4));
TETRIS_LOOKUP.setAction(["RIGHT", "TURN_CCW"], serializePlayerAction(ACTION_SPACE, 5));
TETRIS_LOOKUP.setAction(["LEFT"], serializePlayerAction(ACTION_SPACE, 6));
TETRIS_LOOKUP.setAction(["LEFT", "TURN_CW"], serializePlayerAction(ACTION_SPACE, 7));
TETRIS_LOOKUP.setAction(["LEFT", "TURN_CCW"], serializePlayerAction(ACTION_SPACE, 8));
TETRIS_LOOKUP.setAction(["DOWN"], serializePlayerAction(ACTION_SPACE, 9));
TETRIS_LOOKUP.setAction(["DOWN", "TURN_CW"], serializePlayerAction(ACTION_SPACE, 10));
TETRIS_LOOKUP.setAction(["DOWN", "TURN_CCW"], serializePlayerAction(ACTION_SPACE, 11));

export const TetrisEnvironments = ["tetris/TetrisA-v0"];
export const TetrisControls = ({ sendAction, fps = 30, actorClass, ...props }) => {
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();

  const computeAndSendAction = useCallback(
    (dt) => {
      if (pressedKeys.size === 0 && actorClass === TEACHER_ACTOR_CLASS) {
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
    [pressedKeys, sendAction, actorClass]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div {...props}>
      <div className="flex flex-row gap-1">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
        <FpsCounter className="flex-none w-fit" value={currentFps} />
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right Arrows", "Move piece left/right"],
          ["Down Arrow", "Send piece down"],
          ["w/x", "Turn piece couterclockwise/clockwise"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
