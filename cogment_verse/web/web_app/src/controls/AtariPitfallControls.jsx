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
import { useDocumentKeypressListener, usePressedKeys, useRealTimeUpdate } from "@cogment/cogment-verse-components";
import { createLookup } from "../utils/controlLookup";
import { TEACHER_ACTOR_CLASS } from "../utils/constants";
import { Button, FpsCounter, KeyboardControlList } from "@cogment/cogment-verse-components";
import { serializePlayerAction, TEACHER_NOOP_ACTION, Space } from "../utils/spaceSerialization";

const ACTION_SPACE = new Space({
  discrete: {
    n: 18,
  },
});

// cf. https://www.gymlibrary.ml/environments/atari/#action-space
const ATARI_LOOKUP = createLookup();
ATARI_LOOKUP.setAction([], serializePlayerAction(ACTION_SPACE, 0));
ATARI_LOOKUP.setAction(["FIRE"], serializePlayerAction(ACTION_SPACE, 1));
ATARI_LOOKUP.setAction(["UP"], serializePlayerAction(ACTION_SPACE, 2));
ATARI_LOOKUP.setAction(["RIGHT"], serializePlayerAction(ACTION_SPACE, 3));
ATARI_LOOKUP.setAction(["LEFT"], serializePlayerAction(ACTION_SPACE, 4));
ATARI_LOOKUP.setAction(["DOWN"], serializePlayerAction(ACTION_SPACE, 5));
ATARI_LOOKUP.setAction(["UP", "RIGHT"], serializePlayerAction(ACTION_SPACE, 6));
ATARI_LOOKUP.setAction(["UP", "LEFT"], serializePlayerAction(ACTION_SPACE, 7));
ATARI_LOOKUP.setAction(["DOWN", "RIGHT"], serializePlayerAction(ACTION_SPACE, 8));
ATARI_LOOKUP.setAction(["DOWN", "LEFT"], serializePlayerAction(ACTION_SPACE, 9));
ATARI_LOOKUP.setAction(["UP", "FIRE"], serializePlayerAction(ACTION_SPACE, 10));
ATARI_LOOKUP.setAction(["RIGHT", "FIRE"], serializePlayerAction(ACTION_SPACE, 11));
ATARI_LOOKUP.setAction(["LEFT", "FIRE"], serializePlayerAction(ACTION_SPACE, 12));
ATARI_LOOKUP.setAction(["DOWN", "FIRE"], serializePlayerAction(ACTION_SPACE, 13));
ATARI_LOOKUP.setAction(["UP", "RIGHT", "FIRE"], serializePlayerAction(ACTION_SPACE, 14));
ATARI_LOOKUP.setAction(["UP", "LEFT", "FIRE"], serializePlayerAction(ACTION_SPACE, 15));
ATARI_LOOKUP.setAction(["DOWN", "RIGHT", "FIRE"], serializePlayerAction(ACTION_SPACE, 16));
ATARI_LOOKUP.setAction(["DOWN", "LEFT", "FIRE"], serializePlayerAction(ACTION_SPACE, 17));

export const AtariPitfallEnvironments = ["environments.gym.environment.Environment/ALE/Pitfall-v5"];
export const AtariPitfallControls = ({ sendAction, fps = 30, actorClass, ...props }) => {
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
      }
      if (pressedKeys.has("ArrowDown")) {
        controls.push("DOWN");
      } else if (pressedKeys.has("ArrowUp")) {
        controls.push("UP");
      }
      if (pressedKeys.has(" ")) {
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
      <div className="flex flex-row gap-1">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
        <FpsCounter className="flex-none w-fit" value={currentFps} />
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right/Down/Up Arrows", "Move the character"],
          ["Space", "Jump"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
