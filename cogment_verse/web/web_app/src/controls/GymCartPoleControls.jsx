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
import { cogment_verse } from "../data_pb";
import { useDocumentKeypressListener, usePressedKeys } from "../hooks/usePressedKeys";
import { useRealTimeUpdate } from "../hooks/useRealTimeUpdate";
import { TEACHER_ACTOR_CLASS, TEACHER_NOOP_ACTION } from "../utils/constants";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";

export const GymCartPoleEnvironments = ["environments.gym_adapter.Environment/CartPole-v1"];
export const GymCartPoleControls = ({ sendAction, fps = 30, actorClass, ...props }) => {
  const [paused, setPaused] = useState(true);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();

  const computeAndSendAction = useCallback(
    (dt) => {
      if (pressedKeys.size === 0 && actorClass === TEACHER_ACTOR_CLASS) {
        sendAction(TEACHER_NOOP_ACTION);
        return;
      }

      const action_params = {
        value: {
          properties: [
            {
              discrete: 0,
            },
          ],
        },
      };

      if (pressedKeys.has("ArrowLeft")) {
        action_params.value.properties[0].discrete = 0;
      } else if (pressedKeys.has("ArrowRight")) {
        action_params.value.properties[0].discrete = 1;
      }

      sendAction(new cogment_verse.PlayerAction(action_params));
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
          ["Left/Right Arrows", "push cart to the left/right"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
