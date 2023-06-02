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
import {
  Button,
  createLookup,
  FpsCounter,
  KeyboardControlList,
  serializePlayerAction,
  Space,
  TEACHER_ACTOR_CLASS,
  TEACHER_NOOP_ACTION,
  useDocumentKeypressListener,
  usePressedKeys,
  useRealTimeUpdate,
} from "@cogment/cogment-verse";

const ACTION_SPACE = new Space({
  discrete: {
    n: 6,
  },
});

// cf. https://www.gymlibrary.ml/environments/atari/#action-space
const ATARI_LOOKUP = createLookup();
ATARI_LOOKUP.setAction([], serializePlayerAction(ACTION_SPACE, 0));
ATARI_LOOKUP.setAction(["FIRE"], serializePlayerAction(ACTION_SPACE, 1));
ATARI_LOOKUP.setAction(["UP"], serializePlayerAction(ACTION_SPACE, 2));
ATARI_LOOKUP.setAction(["DOWN"], serializePlayerAction(ACTION_SPACE, 3));
ATARI_LOOKUP.setAction(["RIGHT"], serializePlayerAction(ACTION_SPACE, 4));
ATARI_LOOKUP.setAction(["LEFT"], serializePlayerAction(ACTION_SPACE, 5));

export const AtariPongPzControls = ({ sendAction, fps = 40, actorClass, observation, ...props }) => {
  const [paused, setPaused] = useState(false);
  const [playerPos, setPlayerPos] = useState("left");
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);
  const playerName = observation?.gamePlayerName;
  const playerBox = "opacity-90 py-2 rounded-full items-center text-white font-bold, px-5 text-base outline-none";
  const player1 = `bg-green-500 ${playerBox}`;
  const player2 = `bg-orange-500 ${playerBox}`;
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
      } else if (pressedKeys.has("ArrowUp")) {
        controls.push("UP");
      } else if (pressedKeys.has("Enter")) {
        controls.push("FIRE");
      }
      const action = ATARI_LOOKUP.getAction(controls);
      sendAction(action);
    },
    [pressedKeys, sendAction, actorClass]
  );
  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div {...props}>
      <div className="flex flex-row py-4 gap-1">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
        <FpsCounter className="flex-none w-fit" value={currentFps} />
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right Arrows", "FIRE and MOVE UP/DOWN"],
          ["Up/Down Arrows", "MOVE UP/DOWN"],
          ["Enter", "Fire"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
