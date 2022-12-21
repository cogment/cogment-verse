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
import { TEACHER_ACTOR_CLASS, TEACHER_NOOP_ACTION } from "../utils/constants";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";

// cf. https://www.gymlibrary.ml/environments/atari/#action-space
const ATARI_LOOKUP = createLookup();
ATARI_LOOKUP.setAction([], new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 0 }] } }));
ATARI_LOOKUP.setAction(["FIRE"], new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 1 }] } }));
ATARI_LOOKUP.setAction(["UP"], new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 2 }] } }));
ATARI_LOOKUP.setAction(["DOWN"], new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 3 }] } }));

export const AtariPongPzEnvironments = ["environments.pettingzoo_atari_adapter_v2.Environment/pettingzoo.atari.pong_v3"];
export const AtariPongPzControls = ({ sendAction, fps = 30, actorClass, ...props }) => {
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
          ["Down/Up Arrows", "Move the paddle"],
          ["Spacebar", "Serve the ball"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
