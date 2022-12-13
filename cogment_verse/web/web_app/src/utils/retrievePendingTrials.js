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

export const retrievePendingTrials = async (cogSettings, cogmentOrchestratorWebEndpoint, timeout = 5000) => {
  const context = new Context(cogSettings, "cogment_verse_web");
  const trialController = context.getController(cogmentOrchestratorWebEndpoint);
  let cancelled = false;
  const pendingTrials = new Set();

  return await Promise.race([
    new Promise(async (resolve, reject) => {
      try {
        for await (const trialEvent of trialController.watchTrials()) {
          if (cancelled) {
            console.log("[retrievePendingTrials] watch trials cancelled");
            return;
          }
          if (trialEvent.state === TrialState.PENDING) {
            pendingTrials.add(trialEvent.trialId);
          } else {
            pendingTrials.delete(trialEvent.trialId);
          }
        }
        resolve([...pendingTrials]);
      } catch (error) {
        console.log("erroror", error);
        reject(error);
      }
    }),
    new Promise((resolve) => {
      setTimeout(() => {
        cancelled = true;
        resolve([...pendingTrials]);
      }, timeout);
    }),
  ]);
};
