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
import { useDocumentKeypressListener } from "../hooks/usePressedKeys";
import { useRealTimeUpdate } from "../hooks/useRealTimeUpdate";
import { TEACHER_NOOP_ACTION } from "../utils/constants";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";

export const RealTimeObserverControls = ({ sendAction, fps = 30, ...props }) => {
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const computeAndSendAction = useCallback(
    (dt) => {
      sendAction(TEACHER_NOOP_ACTION);
    },
    [sendAction]
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
      <KeyboardControlList items={[["p", "Pause/Unpause"]]} />
    </div>
  );
};
