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
import { ControlList } from "./ControlList";
import { Countdown } from "./Countdown";
import * as data_pb from "./data_pb";
import { useActions } from "./hooks/useActions";
import { useKeys } from "./hooks/useKeys";
import { useWindowSize } from "./hooks/useWindowSize";

function App() {
  const [pixelData, setPixelData] = useState<Uint8Array | undefined>();
  const [trialStatus, setTrialStatus] = useState("no trial started");
  const [currentTrialId, setCurrentTrialId] = useState<string | undefined>();
  const [countdown, setCountdown] = useState(false);

  const [environmentImplementation, setEnvironmentImplementation] = useState<string>();

  const imgRef = useRef<HTMLImageElement>(null);

  // cogment stuff

  function bufferToBase64(buf: Uint8Array) {
    var binstr = Array.prototype.map
      .call(buf, function (ch) {
        return String.fromCharCode(ch);
      })
      .join("");
    return btoa(binstr);
  }

  const grpcURL = process.env.REACT_APP_GRPCWEBPROXY_URL || "http://localhost:8081";

  type ObservationT = data_pb.cogment_verse.Observation;
  type ActionT = data_pb.cogment_verse.AgentAction;
  type ActorConfigT = data_pb.cogment_verse.HumanConfig;

  useEffect(() => {
    //const canvas = canvasRef.current;
    if (!imgRef) {
      return;
    }
    const img = imgRef.current;
    if (!pixelData || !img) {
      return;
    }

    img.src = "data:image/png;base64," + bufferToBase64(pixelData);
  }, [pixelData]);

  const [event, joinAnyTrial, _sendAction, trialJoined, actorConfig] = useActions<ObservationT, ActionT, ActorConfigT>(
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
    (action: ActionT) => {
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
    if (trialJoined && actorConfig && event.observation && event.observation.pixelData) {
      setPixelData(event.observation.pixelData);
    }
  }, [event, actorConfig, trialJoined]);

  useEffect(() => {
    if (!actorConfig?.environmentSpecs?.implementation) return;
    setEnvironmentImplementation(actorConfig?.environmentSpecs?.implementation);
  }, [actorConfig]);

  const { onKeyDown, onKeyUp } = useKeys(event, sendAction, trialJoined, actorConfig);

  const joinTrial = useCallback(() => {
    setCountdown(false);
    const joinResult = joinAnyTrial();
    if (!joinResult) return;
    if (!joinResult.joinPromise) return;
    setCurrentTrialId(joinResult.trialToJoin);
    setTrialStatus("trial joined");
    joinResult.joinPromise.then(() => {
      setCurrentTrialId(undefined);
      setTrialStatus("waiting to join trial");
      setCountdown(true);
    });
  }, [joinAnyTrial]);

  const windowSize = useWindowSize() as unknown as { width: number; height: number };
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="container" onKeyDown={onKeyDown} onKeyUp={onKeyUp}>
      <div className="cabinet-container">
        <img height={windowSize.height * 2} src={`${process.env.PUBLIC_URL}/assets/Arcade.png`} alt="arcade machine" />
        <div id="screen" className="screen">
          {countdown && <Countdown onAfterCountdown={joinTrial} />}
          {pixelData && (
            <img
              style={{ filter: countdown ? "blur(3px)" : "none" }}
              ref={imgRef}
              className="display"
              tabIndex={0}
              alt="current trial observation"
            />
          )}
        </div>
        <button onClick={joinTrial} className="pushable">
          <span className="front">Join Trial</span>
        </button>
        <div className="status">
          Status: {trialStatus}
          <br></br>
          Trial ID: {currentTrialId}
        </div>
        <div id="expand" className={expanded ? "control-container-open" : "control-container"}>
          <button className="pull-down" onClick={() => setExpanded((expanded) => !expanded)}>
            Controls ≡
          </button>
          <div className="control-group">
            <ControlList environmentImplementation={environmentImplementation} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
