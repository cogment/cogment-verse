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

import React, { useCallback, useState } from "react";
import {
  Button,
  DType,
  FpsCounter,
  Joystick,
  KeyboardControlList,
  OBSERVER_ACTOR_CLASS,
  PlayObserver,
  serializePlayerAction,
  SimplePlay,
  Space,
  TEACHER_ACTOR_CLASS,
  TEACHER_NOOP_ACTION,
  useDocumentKeypressListener,
  useJoystickState,
  usePressedKeys,
  useRealTimeUpdate,
} from "@cogment/cogment-verse";

const ACTION_SPACE = new Space({
  box: {
    low: {
      // Only the dtype and shape of the lower bound are used to serialize the action
      dtype: DType.DTYPE_FLOAT32,
      shape: [2],
    },
  },
});

export const GymLunarLanderContinuousControls = ({ sendAction, fps = 20, actorParams, ...props }) => {
  const actorClassName = actorParams?.className;

  const isTeacher = actorClassName === TEACHER_ACTOR_CLASS;
  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();
  const { joystickPosition, isJoystickActive, setJoystickState } = useJoystickState();

  const [isKeyboardActive, setKeyboardActive] = useState(false);

  const computeAndSendAction = useCallback(
    (dt) => {
      if (isJoystickActive) {
        sendAction(serializePlayerAction(ACTION_SPACE, [joystickPosition[1], -joystickPosition[0]]));
        setKeyboardActive(false);
        return;
      }

      let keyboardAction = null;
      const fast = pressedKeys.has("Shift");
      if (pressedKeys.has("ArrowLeft")) {
        keyboardAction = [0, 0];
        keyboardAction[1] = fast ? 1.0 : 0.75;
      } else if (pressedKeys.has("ArrowRight")) {
        keyboardAction = [0, 0];
        keyboardAction[1] = fast ? -1.0 : -0.75;
      }

      if (pressedKeys.has("ArrowDown")) {
        keyboardAction = keyboardAction || [0, 0];
        keyboardAction[0] = fast ? 1 : 0.25;
      }

      if (keyboardAction != null) {
        sendAction(serializePlayerAction(ACTION_SPACE, keyboardAction));
        setKeyboardActive(true);
        setJoystickState([-keyboardAction[1], keyboardAction[0]], false);
        return;
      }

      if (isTeacher) {
        sendAction(TEACHER_NOOP_ACTION);
      } else {
        sendAction(serializePlayerAction(ACTION_SPACE, [0, 0]));
      }
      setKeyboardActive(false);
      setJoystickState([0, 0], false);
    },
    [isTeacher, pressedKeys, joystickPosition, isJoystickActive, setJoystickState, setKeyboardActive, sendAction]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div {...props}>
      <div className="flex flex-row p-5 justify-center">
        <Joystick
          position={joystickPosition}
          active={isJoystickActive}
          onChange={setJoystickState}
          disabled={paused || isKeyboardActive}
        />
      </div>
      <div className="flex flex-row gap-1">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
        <FpsCounter className="flex-none w-fit" value={currentFps} />
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right Arrows", "Fire left/right engine"],
          ["Down Arrow", "Fire the main engine"],
          ["Shift", "Maintain pressed for 100% power"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};

const PlayGymLunarLanderContinuous = ({ actorParams, ...props }) => {
  const actorClassName = actorParams?.className;

  if (actorClassName === OBSERVER_ACTOR_CLASS) {
    return <PlayObserver actorParams={actorParams} {...props} />;
  }
  return <SimplePlay actorParams={actorParams} {...props} controls={GymLunarLanderContinuousControls} />;
};

export default PlayGymLunarLanderContinuous;
