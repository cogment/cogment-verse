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

import { useCallback, useEffect, useState } from "react";
import { CountdownCircleTimer } from "react-countdown-circle-timer";
import { Button } from "@cogment/cogment-verse-components";
import { KeyboardControlList } from "@cogment/cogment-verse-components";
import { useDocumentKeypressListener } from "@cogment/cogment-verse-components";
import { serializePlayerAction, Space } from "../shared/utils/spaceSerialization";

const TURN_DURATION_SECS = 5;

const ACTION_SPACE = new Space({
  discrete: {
    n: 6,
  },
});

export const OvercookedTurnBasedEnvironments = ["environments.overcooked_adapter.OvercookedEnvironment/overcooked"];
export const OvercookedTurnBasedControls = ({ sendAction, fps = 1, actorClass, observation, ...props }) => {
  const currentPlayer = observation?.currentPlayer;
  const [paused, setPaused] = useState(false);
  const [waitingForAction, setWaitingForAction] = useState(true);

  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const stepDisabled = paused;

  const [turnKey, setTurnKey] = useState(0);
  useEffect(() => {
    setTurnKey((turnKey) => turnKey + 1);
  }, [currentPlayer]);

  const up = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 0));
      setTurnKey((turnKey) => turnKey + 1);
    }
  }, [sendAction]);
  useDocumentKeypressListener("ArrowUp", up);

  const down = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 1));
      setTurnKey((turnKey) => turnKey + 1);
    }
  }, [sendAction]);
  useDocumentKeypressListener("ArrowDown", down);

  const left = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 3));
      setTurnKey((turnKey) => turnKey + 1);
    }
  }, [sendAction]);
  useDocumentKeypressListener("ArrowLeft", left);

  const right = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 2));
      setTurnKey((turnKey) => turnKey + 1);
    }
  }, [sendAction]);
  useDocumentKeypressListener("ArrowRight", right);

  const interact = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 5));
      setTurnKey((turnKey) => turnKey + 1);
    }
  }, [sendAction]);
  useDocumentKeypressListener("Enter", interact);

  const skip = useCallback(() => {
    if (!stepDisabled) {
      sendAction(serializePlayerAction(ACTION_SPACE, 4));
      setTurnKey((turnKey) => turnKey + 1);
    }
  }, [sendAction]);
  useDocumentKeypressListener("Shift", skip);

  return (
    <div {...props}>
      <div className="flex flex-row gap-1">
        <Button className="flex-1 flex justify-center items-center gap-2" onClick={skip} disabled={stepDisabled}>
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
              onComplete={skip}
            />
          </div>
          <div className="flex-initial">
            {currentPlayer ? `Step to "${observation.currentPlayer}" turn` : "turn timer"}
          </div>
        </Button>
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
      </div>

      <KeyboardControlList
        items={[
          ["Left/Right/Up/Down Arrows", "Move player left/right/up/down"],
          ["Enter", "Interact"],
          ["Shift", "Skip to next time step"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};
