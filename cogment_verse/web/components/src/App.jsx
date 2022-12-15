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

import { useCallback, useEffect, useRef, useState } from "react";
import { cogSettings } from "./CogSettings";
import { Countdown } from "./components/Countdown";
import { useActions } from "./hooks/useActions";
import { RenderedScreen } from "./components/RenderedScreen";
import { Button } from "./components/Button";
import { Controls } from "./controls/Controls";
import { ORCHESTRATOR_WEB_ENDPOINT } from "./utils/constants";

function App() {
  const [trialStatus, setTrialStatus] = useState("no trial started");
  const [countdown, setCountdown] = useState(false);

  const [{ trialId, runId, environment, turnBased }, setTrialInfo] = useState({});

  const [event, joinAnyTrial, _sendAction, trialJoined, actorClass, actorConfig] = useActions(
    cogSettings,
    ORCHESTRATOR_WEB_ENDPOINT
  );

  const actionLock = useRef(false);

  useEffect(() => {
    actionLock.current = false;
  }, [event]);

  const sendAction = useCallback(
    (action) => {
      // use lock to ensure we send exactly one action per tick
      if (actionLock.current) return;
      if (_sendAction) {
        _sendAction(action);
        actionLock.current = true;
      }
    },
    [_sendAction]
  );

  useEffect(() => {
    setTrialInfo((trialInfo = {}) => ({
      ...trialInfo,
      environment: actorConfig?.environmentSpecs?.implementation || undefined,
      turnBased: actorConfig?.environmentSpecs?.turnBased || false,
      runId: actorConfig?.runId || undefined,
    }));
  }, [actorConfig]);
  const joinTrial = useCallback(() => {
    setCountdown(false);
    const joinResult = joinAnyTrial();
    if (!joinResult) return;
    if (!joinResult.joinPromise) return;
    setTrialInfo((trialInfo = {}) => ({
      ...trialInfo,
      trialId: joinResult.trialToJoin,
    }));
    setTrialStatus("trial joined");
    joinResult.joinPromise.then(() => {
      setTrialInfo((trialInfo = {}) => ({
        ...trialInfo,
        trialId: undefined,
      }));
      setTrialStatus("waiting to join trial");
      setCountdown(true);
    });
  }, [joinAnyTrial]);

  return (
    <div className="max-w-screen-md mx-auto min-h-screen">
      <div className="p-2">
        <h1 className="text-xl font-semibold">{environment}</h1>
        <div className="text-base">
          {runId}/{trialId}
        </div>
      </div>

      <RenderedScreen
        observation={event.observation}
        overlay={
          countdown ? (
            <Countdown onAfterCountdown={joinTrial} />
          ) : trialId == null ? (
            <Button onClick={joinTrial}>Join Trial</Button>
          ) : null
        }
      />
      <div className="p-2 flex flex-col gap-2">
        {trialJoined ? (
          <Controls
            actorClass={actorClass}
            environment={environment}
            sendAction={sendAction}
            turnBased={turnBased}
            observation={event.observation}
          />
        ) : null}
        <div className="font-mono text-right text-xs lowercase">Status: {trialStatus}</div>
      </div>
    </div>
  );
}

export default App;
