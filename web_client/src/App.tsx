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


  const [event, joinTrial, _sendAction, reset, trialJoined, watchTrials, trialStateList, AgentConfig] = useActions<
    ObservationT,
    ActionT,
    HumanConfigT
  >(
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
    if (trialJoined && AgentConfig && event.observation && event.observation.pixelData) {
      setPixelData(event.observation.pixelData);
    }
  }, [event, AgentConfig, trialJoined]);

  useEffect(() => {

    if (
      trialJoined &&
      AgentConfig &&
      AgentConfig.environmentSpecs &&
      event.observation &&
      event.observation.pixelData
    ) {
      if (!event.last) {
        const action = new data_pb.cogment_verse.AgentAction();
        let action_int = -1;

        if (AgentConfig.role === data_pb.cogment_verse.HumanRole.TEACHER) {
          let keymap = undefined;
          if (AgentConfig.environmentSpecs.implementation) {
            keymap = get_keymap(AgentConfig.environmentSpecs.implementation);
          }

          if (keymap === undefined) {
            console.log(`no keymap defined for actor config ${AgentConfig}`);
          } else {
            for (let item of keymap.action_map) {
              const keySet = new Set<string>(item.keys);
              if (setIsEndOfArray(keySet, pressedKeys)) {
                action_int = item.id;
                break;
              }
            }
          }
        }

        action.discreteAction = action_int;
        if (sendAction) {
          sendAction(action);
        }

        const minDelta = 1000.0 / 30.0; // 30 fps
        const currentTime = new Date().getTime();
        const timeout = lastTime ? Math.max(0, minDelta - (currentTime - lastTime) - 1) : 0;

        if (!timer) {
          setTimer(
            setTimeout(() => {
              const currentTime = new Date().getTime();
              const fps = 1000.0 / Math.max(1, currentTime - lastTime);
              setEmaFps(fpsEmaWeight * fps + (1 - fpsEmaWeight) * emaFps);

              if (sendAction) {
                sendAction(action);
              }
              setTimer(undefined);
              setLastTime(currentTime);
            }, timeout)
          );
        }
      }
    }
    return () => {
      if (timer) {
        clearTimeout(timer);
        setTimer(undefined);
      }
    };
  }, [event, sendAction, trialJoined, AgentConfig, pressedKeys, lastTime, timer, emaFps, fpsEmaWeight]);

  useEffect(() => {
    if (trialJoined) {
      if (AgentConfig && AgentConfig.role === data_pb.cogment_verse.HumanRole.TEACHER) {
        setTrialStatus("trial joined as teacher");
      } else if (AgentConfig && AgentConfig.role === data_pb.cogment_verse.HumanRole.OBSERVER) {
        setTrialStatus("trial joined as observer");
      } else {
        setTrialStatus("trial joined (unknown role)");
      }
      setCanStartTrial(false);
      if (AgentConfig && AgentConfig.environmentSpecs && AgentConfig.environmentSpecs.implementation) {
        setEnvironmentImplementation(AgentConfig.environmentSpecs.implementation);
      }
    } else {
      setTrialStatus("no trial running");
      setCurrentTrialId(undefined);
      setCanStartTrial(true);
    }
  }, [trialJoined, trialStatus, AgentConfig]);

  useEffect(() => {
    if (watchTrials && !watching) {
      watchTrials();
      setWatching(true);
    }
  }, [watchTrials, watching]);

  //This will start a trial as soon as we're connected to the orchestrator
  const triggerJoinTrial = () => {
    if (!joinTrial || trialJoined) {
      return;
    }
    reset();

    if (trialStateList === undefined) {
      return;
    }

    let trialIdToJoin: string | undefined = undefined;

    // find a trial to join
    for (let trialId of Array.from(trialStateList.keys())) {
      let state = trialStateList.get(trialId);

      // trial is pending
      if (state === 2) {
        trialIdToJoin = trialId;
        break;
      }
    }

    if (trialIdToJoin === undefined) {
      console.log("no trial to join");
      return;
    } else {
      console.log(`attempting to join trial ${trialIdToJoin}`);
    }

    if (canStartTrial) {
      //startTrial(trialConfig);
      joinTrial(trialIdToJoin);
      console.log("calling joinTrial");
      setCurrentTrialId(trialIdToJoin);
      if (trialJoined) {
        setCanStartTrial(false);
        setCurrentTrialId(trialIdToJoin);
        console.log("trial joined");
      } else {
        setTrialStatus("could not start trial");
      }
    }
  };

  useEffect(() => {
    const timer = setInterval(triggerJoinTrial, 200);
    return () => {
      clearInterval(timer);
    };
  });


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
