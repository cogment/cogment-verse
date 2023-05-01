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
import {
  useDocumentKeypressListener,
  usePressedKeys,
} from "../../../cogment_verse/web/web_app/src/hooks/usePressedKeys";
import { useRealTimeUpdate } from "../../../cogment_verse/web/web_app/src/hooks/useRealTimeUpdate";
import { TEACHER_ACTOR_CLASS } from "../../../cogment_verse/web/web_app/src/utils/constants";
import { Button } from "../../../cogment_verse/web/web_app/src/components/Button";
import { FpsCounter } from "../../../cogment_verse/web/web_app/src/components/FpsCounter";
import { KeyboardControlList } from "../../../cogment_verse/web/web_app/src/components/KeyboardControlList";
import {
  serializePlayerAction,
  TEACHER_NOOP_ACTION,
  Space,
} from "../../../cogment_verse/web/web_app/src/utils/spaceSerialization";

const ACTION_SPACE = new Space({
  discrete: {
    n: 2,
  },
});

export const Environments = ["environments.gym.environment.Environment/CartPole-v1"];
export const Controls = ({ sendAction, fps = 30, actorClass, ...props }) => {
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

      if (pressedKeys.has("ArrowLeft")) {
        sendAction(serializePlayerAction(ACTION_SPACE, 0));
        return;
      } else if (pressedKeys.has("ArrowRight")) {
        sendAction(serializePlayerAction(ACTION_SPACE, 1));
        return;
      }

      // Default action is left
      sendAction(serializePlayerAction(ACTION_SPACE, 0));
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
