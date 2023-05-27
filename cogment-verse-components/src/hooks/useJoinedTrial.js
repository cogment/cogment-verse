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

import { Context } from "@cogment/cogment-js-sdk";
import { useEffect, useRef, useState } from "react";
import { WEB_ACTOR_NAME, PLAYER_ACTOR_CLASS } from "../constants";

export const TRIAL_STATUS = {
  JOINING: "JOINING",
  ONGOING: "ONGOING",
  ENDED: "ENDED",
  ERROR: "ERROR",
};

export const useJoinedTrial = (cogSettings, cogmentOrchestratorWebEndpoint, trialId, timeout = 5000) => {
  const [[status, error], setTrialStatus] = useState([TRIAL_STATUS.JOINING, null]);
  const [event, setEvent] = useState({
    observation: undefined,
    message: undefined,
    reward: undefined,
    last: false,
    tickId: 0,
  });
  const [sendAction, setSendAction] = useState();
  const [actorParams, setActorParams] = useState(null);

  const actionLock = useRef(false);

  useEffect(() => {
    setTrialStatus([TRIAL_STATUS.JOINING, null]);
    const context = new Context(cogSettings, "cogment_verse_web");

    const timeoutId = setTimeout(() => {
      const error = new Error("Joined trial didn't start actor after timeout");
      console.error(`Error while running trial [${trialId}]`, error);
      setTrialStatus([TRIAL_STATUS.ERROR, error]);
    }, timeout);

    const actorImplementation = async (actorSession) => {
      try {
        if (actorSession.getTrialId() !== trialId) {
          throw new Error(
            `Unexpected error, joined trial [${actorSession.getTrialId()}] doesn't match desired trial [${trialId}]`
          );
        }
        setActorParams({
          name: actorSession.name, // WEB_ACTOR_NAME
          config: actorSession.config,
          className: actorSession.className,
        });

        setSendAction(() => (action) => {
          // use lock to ensure we send exactly one action per tick
          if (actionLock.current) {
            console.warn(
              `trial [${actorSession.getTrialId()}] at tick [${actorSession.getTickId()}] received a 2nd action, ignoring it.`
            );
            return;
          }
          actorSession.doAction(action);
          actionLock.current = true;
        });
        actionLock.current = false;

        actorSession.start();
        clearTimeout(timeoutId);
        setTrialStatus([TRIAL_STATUS.ONGOING, null]);

        let tickId = actorSession.getTickId();
        for await (const { observation, messages, rewards, type } of actorSession.eventLoop()) {
          let nextEvent = {
            observation,
            message: messages[0],
            reward: rewards[0],
            last: type === 3,
            tickId: actorSession.getTickId(),
          };

          const newTick = nextEvent.tickId !== tickId;
          setEvent(nextEvent);
          // Unlock when a new tick starts
          if (newTick) {
            actionLock.current = false;
          }

          tickId = nextEvent.tickId;
        }

        setTrialStatus([TRIAL_STATUS.ENDED, null]);
      } catch (error) {
        setTrialStatus([TRIAL_STATUS.ERROR, error]);
        console.error(`Error while running trial [${trialId}]`, error);
        throw error; // Rethrowing for the sdk to do its job
      }
    };

    context.registerActor(
      actorImplementation,
      WEB_ACTOR_NAME,
      PLAYER_ACTOR_CLASS // actually what we should do is [TEACHER_ACTOR_CLASS, PLAYER_ACTOR_CLASS, OBSERVER_ACTOR_CLASS]
    );

    context.joinTrial(trialId, cogmentOrchestratorWebEndpoint, WEB_ACTOR_NAME).catch((error) => {
      setTrialStatus([TRIAL_STATUS.ERROR, error]);
      console.error(`Error while running trial [${trialId}]`, error);
    });

    return () => clearTimeout(timeoutId);
  }, [cogSettings, cogmentOrchestratorWebEndpoint, trialId, timeout]);

  return [status, actorParams, event, sendAction, error];
};
