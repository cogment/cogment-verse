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

import { Context, TrialState } from "@cogment/cogment-js-sdk";
import { useEffect, useState } from "react";

const sleep = async (delayMs) => new Promise((resolve) => setTimeout(resolve, delayMs));

export const useActiveTrials = (cogSettings, cogmentOrchestratorWebEndpoint, retryDelayMs = 500) => {
  const [trials, setTrials] = useState([]);

  useEffect(() => {
    const context = new Context(cogSettings, "cogment_verse_web");
    const trialController = context.getController(cogmentOrchestratorWebEndpoint);

    let cancelled = false;

    const watchTrials = async (retryNum) => {
      const watchTrialsGenerator = trialController.watchTrials();

      const delayMs = retryNum * retryDelayMs;
      try {
        for await (const trialEvent of watchTrialsGenerator) {
          if (cancelled) {
            console.log("[useActiveTrials] watch trials cancelled");
            return;
          }
          if (trialEvent.state === TrialState.ENDED) {
            setTrials((trials) => trials.filter(({ trialId }) => trialId !== trialEvent.trialId));
            continue;
          }
          setTrials((trials) => {
            const trialIdx = trials.findIndex(({ trialId }) => trialId === trialEvent.trialId);
            if (trialIdx === -1) {
              return [...trials, { trialId: trialEvent.trialId, state: trialEvent.state }];
            }
            if (trials[trialIdx].state !== trialEvent.state) {
              trials[trialIdx].state = trialEvent.state;
              return [...trials];
            }
            return trials;
          });
        }
        console.warn(`[useActiveTrials] watch trials returned early, restarting in ${delayMs}ms`);
      } catch (error) {
        console.warn(`[useActiveTrials] error during watch trials (${error}), restarting in ${delayMs}ms`);
      }
      await sleep(delayMs);
      await watchTrials(retryNum + 1);
    };

    watchTrials(0);

    return () => {
      cancelled = true;
    };
  }, [cogSettings, cogmentOrchestratorWebEndpoint, retryDelayMs]);

  return trials;
};
