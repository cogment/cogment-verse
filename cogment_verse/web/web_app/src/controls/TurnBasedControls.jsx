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

import { useCallback, useEffect, useState } from "react";
import { CountdownCircleTimer } from "react-countdown-circle-timer";
import { Button } from "../../components/Button";
import { DPad, DPAD_BUTTONS, usePressedButtons } from "../../components/DPad";
import { FpsCounter } from "../../components/FpsCounter";
import { KeyboardControlList } from "../../components/KeyboardControlList";
import { useDocumentKeypressListener, usePressedKeys } from "../../hooks/usePressedKeys";
import { useRealTimeUpdate } from "../../hooks/useRealTimeUpdate";
import { serializePlayerAction, Space, TEACHER_NOOP_ACTION } from "../../utils/spaceSerialization";


const ACTION_SPACE = new Space({
    discrete: {
      n: 4,
    },
  });


export const AdaptiveGridTurnBasedEnvironments = [
  "environments.adaptive_grid.adaptive_grid_adapter.AdaptiveGridEnvironment/adaptive_grid_turn_based"
];

export const AdaptiveGridTurnBasedControls = ({ sendAction, fps = 30, actorClass, observation, ...props }) => {
  const currentPlayer = observation?.currentPlayer;
  const [turnKey, setTurnKey] = useState(0);
  const [paused, setPaused] = useState(false);
  const [waitingForAction, setWaitingForAction] = useState(true);
  const stepDisabled = !waitingForAction || paused;

  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const left = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 0));
    }
  }, [sendAction]);

  const right = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 1));
    }
  }, [sendAction]);

  const up = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 2));
    }
  }, [sendAction]);

  const down = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 3));
    }
  }, [sendAction]);

  useDocumentKeypressListener("ArrowLeft", left);
  useDocumentKeypressListener("ArrowRight", right);
  useDocumentKeypressListener("ArrowUp", up);
  useDocumentKeypressListener("ArrowDown", down);


  return (
    <div {...props}>
      <div className="flex flex-row gap-1">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right/Up/Down Arrows", "Move end-effector left/right/up/down"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
