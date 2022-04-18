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

import { Reward } from "@cogment/cogment-js-sdk";
import { CogMessage } from "@cogment/cogment-js-sdk/dist/cogment/types/CogMessage";
import { useEffect, useState } from "react";
import data_pb from "../data_pb";
import { get_keymap, useControls } from "./useControls";
const fpsEmaWeight = 1 / 60.0;

type Event<ObservationT> = {
  observation?: ObservationT;
  message?: CogMessage;
  reward?: Reward;
  last: boolean;
  tickId: number;
};
type SendAction<ActionT> = (action: ActionT) => void;

const setContains = <T extends unknown>(A: Set<T>, B: Set<T>): boolean => {
  if (A.size > B.size) {
    return false;
  }

  for (let a of A) {
    if (!B.has(a)) {
      return false;
    }
  }
  return true;
};

const setEquals = <T extends unknown>(A: Set<T>, B: Set<T>): boolean => {
  return setContains(A, B) && setContains(B, A);
};

const setIsEndOfArray = <T extends unknown>(A: Set<T>, B: T[]): boolean => {
  for (let i = 1; i <= B.length; i++) {
    const testSet = new Set(B.slice(i * -1));
    if (setEquals(A, testSet)) return true;
  }
  return false;
};

export const useKeys = (
  event: Event<data_pb.cogment_verse.Observation>,
  sendAction: SendAction<data_pb.cogment_verse.AgentAction>,
  trialJoined: boolean,
  actorConfig?: data_pb.cogment_verse.HumanConfig
) => {
  const [pressedKeys, onKeyDown, onKeyUp] = useControls();
  const [lastTime, setLastTime] = useState<DOMHighResTimeStamp>(0);
  const [emaFps, setEmaFps] = useState(0.0);
  const [timer, setTimer] = useState<NodeJS.Timeout>();

  useEffect(() => {
    if (
      trialJoined &&
      actorConfig?.environmentSpecs?.implementation &&
      event.observation &&
      event.observation.pixelData
    ) {
      if (!event.last) {
        const action = new data_pb.cogment_verse.AgentAction();
        let action_int = -1;
        const keymap = get_keymap(actorConfig.environmentSpecs?.implementation);

        if (keymap === undefined) {
          console.log(`no keymap defined for actor config ${actorConfig.environmentSpecs?.implementation}`);
        } else {
          for (let item of keymap.action_map) {
            const keySet = new Set<string>(item.keys);
            if (setIsEndOfArray(keySet, pressedKeys)) {
              action_int = item.id;
              break;
            }
          }
        }

        action.discreteAction = action_int;

        const targetDelta = 1000.0 / 30.0; // 30 fps
        const currentTime = new Date().getTime();
        const timeout = lastTime ? Math.max(0, targetDelta - (currentTime - lastTime)) : 0;

        if (!timer) {
          setTimer(
            setTimeout(() => {
              const currentTime = new Date().getTime();
              const fps = 1000.0 / Math.max(1, currentTime - lastTime);
              setEmaFps(fpsEmaWeight * fps + (1 - fpsEmaWeight) * emaFps);

              if (sendAction) {
                sendAction(action);
              }
              setTimer(undefined);
              setLastTime(currentTime);
            }, timeout)
          );
        }
      }
    }
    return () => {
      if (timer) {
        clearTimeout(timer);
        setTimer(undefined);
      }
    };
  }, [event, sendAction, trialJoined, actorConfig, pressedKeys, lastTime, timer, emaFps]);
  return { onKeyDown, onKeyUp };
};
