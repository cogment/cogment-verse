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

import { useCallback } from "react";
import { cogSettings } from "../CogSettings";
import { Countdown } from "../components/Countdown";
import { Button } from "../components/Button";
import { useJoinedTrial, TRIAL_STATUS } from "../hooks/useJoinTrial";
import { RenderedScreen } from "../components/RenderedScreen";
import { Controls } from "../controls/Controls";
import { ORCHESTRATOR_WEB_ENDPOINT } from "../utils/constants";
import { useParams, useNavigate } from "react-router-dom";

const BETWEEN_TIMEOUT = 1000;
const JOIN_TIMEOUT = 5000;

const ErrorCard = ({ error }) => (
  <div className="border-l-8 border-red-600 bg-white rounded p-5 shadow-md">
    <h2 className="text-xl font-semibold mb-2">Oups...</h2>
    <p>{error.message}</p>
    {error.cause ? (
      <p>
        <span className="font-semibold">cause: </span>
        {error.cause.message}
      </p>
    ) : null}
    <Button className="mt-2" to="/" reloadDocument>
      Back to trial selection
    </Button>
  </div>
);

const Play = () => {
  const { trialId } = useParams();
  const navigate = useNavigate();
  const [trialStatus, actorParams, event, sendAction, trialError] = useJoinedTrial(
    cogSettings,
    ORCHESTRATOR_WEB_ENDPOINT,
    trialId,
    JOIN_TIMEOUT
  );

  const environment = actorParams?.config?.environmentSpecs?.implementation || undefined;
  const turnBased = actorParams?.config?.environmentSpecs?.turnBased || false;
  const runId = actorParams?.config?.runId;
  const actorClassName = actorParams?.className;

  const redirectToPlayAny = useCallback(() => navigate(".."), [navigate]);

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
          trialStatus === TRIAL_STATUS.JOINING ? (
            <Countdown duration={JOIN_TIMEOUT} />
          ) : trialStatus === TRIAL_STATUS.ENDED ? (
            <Countdown onAfterCountdown={redirectToPlayAny} duration={BETWEEN_TIMEOUT} />
          ) : trialStatus === TRIAL_STATUS.ERROR ? (
            <ErrorCard error={trialError} />
          ) : null
        }
      />
      <div className="p-2 flex flex-col gap-2">
        {trialStatus === TRIAL_STATUS.ONGOING ? (
          <Controls
            actorClass={actorClassName}
            environment={environment}
            sendAction={sendAction}
            turnBased={turnBased}
            observation={event.observation}
          />
        ) : null}
      </div>
    </div>
  );
};

export default Play;
