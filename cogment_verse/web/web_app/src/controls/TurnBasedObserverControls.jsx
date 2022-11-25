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
import { useDocumentKeypressListener } from "../hooks/usePressedKeys";
import { TEACHER_NOOP_ACTION } from "../utils/constants";
import { Button } from "../components/Button";
import { KeyboardControlList } from "../components/KeyboardControlList";
import { CountdownCircleTimer } from "react-countdown-circle-timer";

const TURN_DURATION_SECS = 10;

export const TurnBasedObserverControls = ({ sendAction, observation, actorClass, ...props }) => {
  const currentPlayer = observation?.currentPlayer;

  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const [expectingAction, setExpectingAction] = useState(true);
  const stepDisabled = !expectingAction || paused;
  const step = useCallback(() => {
    if (!stepDisabled) {
      sendAction(TEACHER_NOOP_ACTION);
      setExpectingAction(false);
    }
  }, [sendAction, stepDisabled]);
  useDocumentKeypressListener(" ", step);

  const [turnKey, setTurnKey] = useState(0);
  useEffect(() => {
    setTurnKey((turnKey) => turnKey + 1);
    setExpectingAction(true);
  }, [currentPlayer]);

  return (
    <div {...props}>
      <div className="flex flex-row gap-1">
        <Button className="flex-1 flex justify-center items-center gap-2" onClick={step} disabled={stepDisabled}>
          <div className="flex-initial">
            <CountdownCircleTimer
              size={20}
              strokeWidth={5}
              strokeLinecap="square"
              key={turnKey}
              duration={TURN_DURATION_SECS}
              colors="#fff"
              trailColor="#555"
              isPlaying={!paused}
              onComplete={step}
            />
          </div>
          <div className="flex-initial">{currentPlayer ? `Step to "${observation.currentPlayer}" turn` : "blop"}</div>
        </Button>
      </div>
      <KeyboardControlList
        items={[
          ["p", "Pause/Unpause"],
          ["space", "step"],
        ]}
      />
    </div>
  );
};
