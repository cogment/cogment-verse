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

import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { cogSettings } from "./CogSettings";
import { Countdown } from "./components/Countdown";
import { useActions } from "./hooks/useActions";
import { Layout } from "./components/Layout";
import { RenderedScreen } from "./components/RenderedScreen";
import { Controls } from "./controls/Controls";

function App() {
  const [trialStatus, setTrialStatus] = useState("no trial started");
  const [countdown, setCountdown] = useState(false);

  const [{ trialId, runId, environment, role }, setTrialInfo] = useState({});

  const grpcURL = process.env.REACT_APP_ORCHESTRATOR_HTTP_ENDPOINT || "http://localhost:8081";

  const [event, joinAnyTrial, _sendAction, trialJoined, actorConfig] = useActions(
    cogSettings,
    "web_actor", // actor name
    "teacher_agent", // actor class
    grpcURL
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
      runId: actorConfig?.runId || undefined,
      role: actorConfig?.role,
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
    <Layout>
      <h1>{environment}</h1>
      <div>
        {runId}/{trialId}
      </div>
      <RenderedScreen
        observation={event.observation}
        overlay={
          countdown ? (
            <Countdown onAfterCountdown={joinTrial} />
          ) : trialId == null ? (
            <button onClick={joinTrial}>Join Trial</button>
          ) : null
        }
      />
      <div>Status: {trialStatus}</div>
      {trialJoined ? <Controls role={role} environment={environment} sendAction={sendAction} /> : null}
    </Layout>
  );
}

export default App;
