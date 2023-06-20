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

import { useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  DynamicallyImportedComponent,
  JOIN_TRIAL_TIMEOUT,
  OBSERVER_ACTOR_CLASS,
  ORCHESTRATOR_WEB_ENDPOINT,
  DEFAULT_SPEC_TYPE,
  PlayObserver,
  SimplePlay,
  WEB_BASE_URL,
} from "../../shared";
import { useJoinedTrial } from "../hooks/useJoinedTrial";
import { cogSettings } from "../../CogSettings";

const Play = () => {
  const { trialId } = useParams();
  const navigate = useNavigate();
  const redirectToPlayAny = useCallback(() => navigate(".."), [navigate]);

  const [trialStatus, actorParams, event, sendAction, trialError] = useJoinedTrial(
    cogSettings,
    ORCHESTRATOR_WEB_ENDPOINT,
    trialId,
    JOIN_TRIAL_TIMEOUT
  );

  if (actorParams == null) {
    return (
      <SimplePlay
        trialId={trialId}
        trialStatus={trialStatus}
        actorParams={actorParams}
        event={event}
        sendAction={sendAction}
        trialError={trialError}
        onNextTrial={redirectToPlayAny}
      />
    );
  }

  const specType = actorParams?.config?.specType;

  console.log("currentPlayer spec_type: " + specType)
  console.log("environmentSpecs: " + actorParams?.config?.environmentSpecs?.actorSpecs)

  const actorSpecs = actorParams?.config?.environmentSpecs?.actorSpecs?.find((spec) => spec.specType === specType) || undefined;
  const implementation = actorParams?.config?.environmentSpecs?.implementation || undefined;
  const componentFile = actorSpecs?.webComponentsFile || undefined;

  console.log("actorSpecs: " + actorSpecs)

  console.log("environmentSpecs.specType: " + actorSpecs?.specType)
  console.log("implementation: " + implementation)
  console.log("componentFile: " + componentFile)

  if (implementation != null && componentFile != null) {
    return (
      <DynamicallyImportedComponent
        moduleUrl={`${WEB_BASE_URL}/components/environments/${implementation}/${componentFile}`}
        trialId={trialId}
        trialStatus={trialStatus}
        actorParams={actorParams}
        event={event}
        sendAction={sendAction}
        trialError={trialError}
        onNextTrial={redirectToPlayAny}
      />
    );
  }

  const actorClassName = actorParams?.className;

  if (actorClassName === OBSERVER_ACTOR_CLASS) {
    return (
      <PlayObserver
        trialId={trialId}
        trialStatus={trialStatus}
        actorParams={actorParams}
        event={event}
        sendAction={sendAction}
        trialError={trialError}
        onNextTrial={redirectToPlayAny}
      />
    );
  } else throw new Error(`No web component defined for environment "${implementation}"`);
};

export default Play;
