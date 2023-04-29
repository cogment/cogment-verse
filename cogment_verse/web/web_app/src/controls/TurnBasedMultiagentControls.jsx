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


const TURN_DURATION_SECS = 2;
const ACTION_SPACE = new Space({
    discrete: {
      n: 4,
    },
  });


export const AdaptiveGridTurnBasedMultiplayerEnvironments = [
  "environments.adaptive_grid.adaptive_grid_adapter_multiplayer.AdaptiveGridEnvironment/adaptive_grid_turn_based"
];

export const AdaptiveGridTurnBasedMultiplayerControls = ({ sendAction, fps = 30, actorClass, observation, ...props }) => {
  const currentPlayer = observation?.currentPlayer;
  const [turnKey, setTurnKey] = useState(0);
  const [paused, setPaused] = useState(false);
  const [waitingForAction, setWaitingForAction] = useState(true);
  const stepDisabled = !waitingForAction || paused;
  const timerDisabled = waitingForAction || paused;

  console.log("currentPlayer: " + currentPlayer)

  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const step = useCallback(() => {
    if (!waitingForAction & !paused) {
      sendAction(TEACHER_NOOP_ACTION);

      setTurnKey((turnKey) => turnKey + 1);
      setWaitingForAction(true);
    }
  }, [sendAction, stepDisabled]);
  useDocumentKeypressListener(" ", step);

  const left = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 0));

      setTurnKey((turnKey) => turnKey + 1);
      setWaitingForAction(false);
    }
  }, [sendAction]);

  const right = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 1));

      setTurnKey((turnKey) => turnKey + 1);
      setWaitingForAction(false);
    }
  }, [sendAction]);

  const up = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 2));

      setTurnKey((turnKey) => turnKey + 1);
      setWaitingForAction(false);
    }
  }, [sendAction]);

  const down = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 3));

      setTurnKey((turnKey) => turnKey + 1);
      setWaitingForAction(false);
    }
  }, [sendAction]);

  useDocumentKeypressListener("ArrowLeft", left);
  useDocumentKeypressListener("ArrowRight", right);
  useDocumentKeypressListener("ArrowUp", up);
  useDocumentKeypressListener("ArrowDown", down);


  return (
    <div {...props}>
      <div className="flex flex-row gap-1">
        <Button className="flex-1 flex justify-center items-center gap-2" onClick={step} disabled={timerDisabled}>
            <div className="flex-initial">
              <CountdownCircleTimer
                size={20}
                strokeWidth={5}
                strokeLinecap="square"
                key={turnKey}
                duration={TURN_DURATION_SECS}
                colors="#fff"
                trailColor="#555"
                isPlaying={!timerDisabled}
                onComplete={step}
              />
            </div>
            <div className="flex-initial">{currentPlayer ? `${observation.currentPlayer} turn` : "blop"}</div>
          </Button>
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right/Up/Down Arrows", "Move end-effector left/right/up/down"],
          ["Space", "Skip timer to next step"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
