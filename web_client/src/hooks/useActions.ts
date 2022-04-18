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

import { CogSettings, Context, MessageBase, Reward } from "@cogment/cogment-js-sdk";
import { CogMessage } from "@cogment/cogment-js-sdk/dist/cogment/types/CogMessage";
import { useCallback, useEffect, useState } from "react";
export type SendAction<ActionT> = (action: ActionT) => void;

type JoinTrial = (trialId: string) => false | Promise<void>;
type JoinAnyTrial = () =>
  | false
  | {
      trialToJoin: string;
      joinPromise: false | Promise<void>;
    };
export type Event<ObservationT> = {
  observation?: ObservationT;
  message?: CogMessage;
  reward?: Reward;
  last: boolean;
  tickId: number;
};

export type TrialStates = { [trialId: string]: number };

export type Policy<ObservationT, ActionT> = (event: Event<ObservationT>) => ActionT;
export type WatchTrials = () => void;
export type UseActions = <ObservationT, ActionT extends MessageBase, ActorConfigT>(
  _cogSettings: CogSettings,
  actorName: string,
  actorClass: string,
  grpcURL: string
) => [
  event: Event<ObservationT>,
  joinAnyTrial: JoinAnyTrial,
  sendAction: SendAction<ActionT> | undefined,
  trialJoined: boolean,
  actorConfig: ActorConfigT | undefined
];

export const useActions: UseActions = <ObservationT, ActionT extends MessageBase, ActorConfigT>(
  cogSettings: CogSettings,
  actorName: string,
  actorClass: string,
  grpcURL: string
) => {
  type EventT = Event<ObservationT>;

  const [event, setEvent] = useState<EventT>({
    observation: undefined,
    message: undefined,
    reward: undefined,
    last: false,
    tickId: 0,
  });

  const [trialStates, setTrialStates] = useState<TrialStates>({});
  const [trialJoined, setTrialJoined] = useState(false);

  const [joinTrial, setJoinTrial] = useState<JoinTrial>();
  const [sendAction, setSendAction] = useState<SendAction<ActionT>>();

  const [watchTrials, setWatchTrials] = useState<WatchTrials>();

  const [actorConfig, setActorConfig] = useState<any>();

  //Set up the connection and register the actor only once, regardless of re-rendering
  useEffect(() => {
    const actor = { name: actorName, actorClass: actorClass };

    const context = new Context<ActionT, ObservationT>(cogSettings, actorName);

    context.registerActor(
      async (actorSession) => {
        let tickId = 0;

        actorSession.start();

        // todo: figure out why this cast is necessary (wrong template argument somewhere?)
        setActorConfig(actorSession.config as ActorConfigT);

        //Double arrow function here beause react will turn a single one into a lazy loaded function
        setSendAction(() => (action: ActionT) => {
          actorSession.doAction(action);
        });

        for await (const { observation, messages, rewards, type } of actorSession.eventLoop()) {
          //Parse the observation into a regular JS object
          //TODO: this will eventually be part of the API

          let observationOBJ = observation && (observation as ObservationT | undefined);

          let next_event = {
            observation: observationOBJ,
            message: messages[0],
            reward: rewards[0],
            last: type === 3,
            tickId: tickId++,
          };

          setEvent(next_event);

          if (next_event.last) {
            break;
          }
        }
      },
      actor.name,
      actor.actorClass
    );

    //Creating the trial controller must happen after actors are registered
    const trialController = context.getController(grpcURL);

    setJoinTrial(() => (trialId: string) => {
      try {
        setTrialJoined(true);
        const joinTrialPromise = context.joinTrial(trialId, grpcURL, actor.name).then(() => setTrialJoined(false));
        return joinTrialPromise;
      } catch (error) {
        console.log(`failed to start trial: ${error}`);
        return false;
      }
    });
    setWatchTrials(() => async () => {
      const doWatchTrial = async () => {
        const watchTrialsGenerator = trialController.watchTrials();
        try {
          for await (const trialStateMsg of watchTrialsGenerator) {
            const { trialId, state } = trialStateMsg;
            setTrialStates((trialStates) => {
              const newTrials = { ...trialStates, [trialId]: state };
              return newTrials;
            });
          }
          console.error("watch trials returned early, restarting");
        } catch (error) {
          console.log(`error during watch trials, ${error} restarting`);
        }
        await doWatchTrial();
      }
      await doWatchTrial();
    });
  }, [cogSettings, actorName, actorClass, grpcURL]);

  useEffect(() => {
    if (!watchTrials) return;
    watchTrials();
  }, [watchTrials]);

  const joinAnyTrial = useCallback(() => {
    if (!joinTrial || trialJoined) return false;
    let trialToJoin = Object.keys(trialStates).find((trialId) => trialStates[trialId] === 2);
    if (!trialToJoin) return false;
    const joinPromise = joinTrial(trialToJoin);
    console.log("joining trial", trialToJoin);
    return { trialToJoin, joinPromise };
  }, [joinTrial, trialJoined, trialStates]);

  return [event, joinAnyTrial, sendAction, trialJoined, actorConfig];
};
