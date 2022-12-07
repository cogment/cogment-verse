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

import { TrialState } from "@cogment/cogment-js-sdk";
import { Link } from "react-router-dom";
import { useActiveTrials } from "../hooks/useActiveTrials";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleNotch } from "@fortawesome/free-solid-svg-icons";
import classNames from "classnames";

import { cogSettings } from "../CogSettings";
import { ORCHESTRATOR_WEB_ENDPOINT } from "../utils/constants";

const trialStatePillClassNames = ["rounded-md", "py-2", "px-5", "text-center", "text-sm", "text-white"];
const TrialStatePill = ({ state, trialId, ...otherProps }) => {
  if (state === TrialState.PENDING) {
    return (
      <Link
        className={classNames(trialStatePillClassNames, "bg-emerald-500 hover:bg-emerald-800")}
        to={`/play/${trialId}`}
        {...otherProps}
      >
        <FontAwesomeIcon icon={faCircleNotch} shake /> Join pending trial...
      </Link>
    );
  } else if (state === TrialState.RUNNING) {
    return (
      <span className={classNames(trialStatePillClassNames, "bg-slate-600")} {...otherProps}>
        <FontAwesomeIcon icon={faCircleNotch} spin /> Trial running
      </span>
    );
  } else {
    return (
      <span className={classNames(trialStatePillClassNames, "bg-red-600")} {...otherProps}>
        {state}
      </span>
    );
  }
};

const Trials = () => {
  const trials = useActiveTrials(cogSettings, ORCHESTRATOR_WEB_ENDPOINT);
  return (
    <div>
      <h1 className="text-xl font-semibold mt-5">Active trials</h1>
      <ul className="my-5 flex flex-col">
        {trials.map(({ trialId, state }, index) => (
          <li key={index} className={classNames("flex flex-row items-center p-2", { "bg-slate-100": index % 2 === 0 })}>
            <div className="grow font-medium font-mono">{trialId}</div>
            <TrialStatePill state={state} trialId={trialId} />
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Trials;
