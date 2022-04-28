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
import { useDocumentKeypressListener, usePressedKeys } from "../hooks/usePressedKeys";
import { useRealTimeUpdate } from "../hooks/useRealTimeUpdate";
import { createLookup } from "../utils/controlLookup";
import { TEACHER_NOOP_ACTION } from "../utils/constants";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";

// cf. https://www.gymlibrary.ml/environments/atari/#action-space
const ATARI_LOOKUP = createLookup();
ATARI_LOOKUP.setAction([], new cogment_verse.AgentAction({ discreteAction: 0 }));
ATARI_LOOKUP.setAction(["FIRE"], new cogment_verse.AgentAction({ discreteAction: 1 }));
ATARI_LOOKUP.setAction(["UP"], new cogment_verse.AgentAction({ discreteAction: 2 }));
ATARI_LOOKUP.setAction(["RIGHT"], new cogment_verse.AgentAction({ discreteAction: 3 }));
ATARI_LOOKUP.setAction(["LEFT"], new cogment_verse.AgentAction({ discreteAction: 4 }));
ATARI_LOOKUP.setAction(["DOWN"], new cogment_verse.AgentAction({ discreteAction: 5 }));
ATARI_LOOKUP.setAction(["UP", "RIGHT"], new cogment_verse.AgentAction({ discreteAction: 6 }));
ATARI_LOOKUP.setAction(["UP", "LEFT"], new cogment_verse.AgentAction({ discreteAction: 7 }));
ATARI_LOOKUP.setAction(["DOWN", "RIGHT"], new cogment_verse.AgentAction({ discreteAction: 8 }));
ATARI_LOOKUP.setAction(["DOWN", "LEFT"], new cogment_verse.AgentAction({ discreteAction: 9 }));
ATARI_LOOKUP.setAction(["UP", "FIRE"], new cogment_verse.AgentAction({ discreteAction: 10 }));
ATARI_LOOKUP.setAction(["RIGHT", "FIRE"], new cogment_verse.AgentAction({ discreteAction: 11 }));
ATARI_LOOKUP.setAction(["LEFT", "FIRE"], new cogment_verse.AgentAction({ discreteAction: 12 }));
ATARI_LOOKUP.setAction(["DOWN", "FIRE"], new cogment_verse.AgentAction({ discreteAction: 13 }));
ATARI_LOOKUP.setAction(["UP", "RIGHT", "FIRE"], new cogment_verse.AgentAction({ discreteAction: 14 }));
ATARI_LOOKUP.setAction(["UP", "LEFT", "FIRE"], new cogment_verse.AgentAction({ discreteAction: 15 }));
ATARI_LOOKUP.setAction(["DOWN", "RIGHT", "FIRE"], new cogment_verse.AgentAction({ discreteAction: 16 }));
ATARI_LOOKUP.setAction(["DOWN", "LEFT", "FIRE"], new cogment_verse.AgentAction({ discreteAction: 17 }));

export const AtariPitfallEnvironments = ["atari/Pitfall"];
export const AtariPitfallControls = ({ sendAction, fps = 30, role, ...props }) => {
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
    [pressedKeys, sendAction, role]
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
