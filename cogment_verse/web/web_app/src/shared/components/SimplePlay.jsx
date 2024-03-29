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

import { Countdown } from "./Countdown";
import { Button } from "./Button";
import { RenderedScreen } from "./RenderedScreen";
import { Inspector } from "./Inspector";
import { JOIN_TRIAL_TIMEOUT, BETWEEN_TRIALS_TIMEOUT, TRIAL_STATUS } from "../utils";

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

const DefaultControls = () => null;

export const SimplePlay = ({
  trialId,
  trialStatus,
  actorParams,
  event,
  sendAction,
  trialError,
  onNextTrial,
  controls = DefaultControls,
}) => {
  const Controls = controls;

  return (
    <div className="max-w-screen-md mx-auto min-h-screen justify-center">
      <Inspector trialId={trialId} event={event} actorParams={actorParams} className="my-2" />
      <RenderedScreen
        observation={event.observation}
        overlay={
          trialStatus === TRIAL_STATUS.JOINING ? (
            <Countdown duration={JOIN_TRIAL_TIMEOUT} />
          ) : trialStatus === TRIAL_STATUS.ENDED ? (
            <Countdown onAfterCountdown={onNextTrial} duration={BETWEEN_TRIALS_TIMEOUT} />
          ) : trialStatus === TRIAL_STATUS.ERROR ? (
            <ErrorCard error={trialError} />
          ) : null
        }
      />
      <div className="p-2 flex flex-col gap-2">
        {trialStatus === TRIAL_STATUS.ONGOING ? (
          <Controls
            actorParams={actorParams}
            sendAction={sendAction}
            observation={event.observation}
            tickId={event.tickId}
          />
        ) : null}
      </div>
    </div>
  );
};
