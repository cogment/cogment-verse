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

import { redirect } from "react-router-dom";
import { retrievePendingTrials } from "../utils/retrievePendingTrials";

import { cogSettings } from "../CogSettings";
import { ORCHESTRATOR_WEB_ENDPOINT } from "../utils/constants";

export const loader = async () => {
  const pendingTrials = await retrievePendingTrials(cogSettings, ORCHESTRATOR_WEB_ENDPOINT, 500);
  if (pendingTrials.length === 0) {
    return redirect("/");
  }
  return redirect(pendingTrials[0]);
};
