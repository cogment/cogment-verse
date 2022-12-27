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

import { useCallback, useEffect, useState } from "react";
import { cogment_verse } from "../data_pb";
import { useDocumentKeypressListener, usePressedKeys } from "../hooks/usePressedKeys";
import { useRealTimeUpdateV2 } from "../hooks/useRealTimeUpdate_v2";
import { createLookup } from "../utils/controlLookup";
import { WEB_ACTOR_NAME } from "../utils/constants";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { Switch } from "../components/Switch"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faThumbsUp, faThumbsDown } from "@fortawesome/free-solid-svg-icons";

// cf. https://www.gymlibrary.ml/environments/atari/#action-space
// const ATARI_LOOKUP = createLookup();
// ATARI_LOOKUP.setAction(["NEUTRAL"], new cogment_verse.ObserverAction({ value: 0 }));
// ATARI_LOOKUP.setAction(["LIKE"], new cogment_verse.ObserverAction({ value: 1 }));
// ATARI_LOOKUP.setAction(["DISLIKE"], new cogment_verse.ObserverAction({ value: -1 }));

export const AtariPongPzHfbEnvironments = ["environments.pettingzoo_atari_adapter.HumanFeedbackEnvironment/pettingzoo.atari.pong_v3"];
export const AtariPongPzFeedback = ({ sendAction, fps = 30, actorClass, observation, ...props }) => {
    const [paused, setPaused] = useState(false);
    const [humanMode, setHumanMode] = useState(true);
    const [selectedFeedback, setFeedback] = useState(0);
    const [humanTurn, setHumanTurn] = useState(false);
    const [playerPos, setPlayerPos] = useState('left');
    const playerName = observation?.playerEvaluated;
    const timeStep = observation?.step;
    // const isHumanTurn = observation?.isHuman;

    // useEffect(() => {
    //     if (playerName.includes('first')) {
    //         setPlayerPos('left');
    //     } else {
    //         setPlayerPos('right');
    //     }
    // }, [playerName])

    // useEffect(() => {
    //     if (isHumanTurn) {
    //         setHumanTurn(true);
    //     } else {
    //         setHumanTurn(false);
    //     }
    // }, [isHumanTurn])

    const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
    const toggleHuman = useCallback(() => setHumanMode((humanMode) => !humanMode), [setHumanMode]);
    useDocumentKeypressListener("p", togglePause);
    useDocumentKeypressListener("h", toggleHuman);

    const { currentFps } = useRealTimeUpdateV2(fps, paused);

    if (!humanMode) {
        // Neural action 
        const action = new cogment_verse.ObserverAction({ value: { properties: [{ simpleBox: { values: [0.0] } }] } });
        sendAction(action);
    }
    const sendFeedback = (feedback) => {
        setFeedback(feedback == "LIKE" ? 1 : -1);
        setTimeout(() => { setFeedback(0) }, 150);
        const action = new cogment_verse.ObserverAction({ value: { properties: [{ simpleBox: { values: feedback == "LIKE" ? [1.0] : [-1.0] } }] } });
        sendAction(action);
    };

    return (
        <div {...props}>
            <div className={playerPos == 'right' ? "flex flex-row-reverse text-bold text-base" : "flex flex-col-reverse text-bold text-base"}>
                {timeStep}
            </div>
            {humanMode && <div className="text-center" style={{
                paddingBottom: 10,
                padingTop: 10,
            }}>
                <button
                    type="button"
                    onClick={() => sendFeedback("LIKE")}
                // disabled={selectedFeedback != 0}
                >
                    <FontAwesomeIcon
                        icon={faThumbsUp}
                        style={{ paddingRight: 10 }}
                        size="4x"
                        color={selectedFeedback == 1 ? "green" : "gray"}
                    />
                </button>
                <button
                    type="button"
                    onClick={() => sendFeedback("DISLIKE")}
                // disabled={selectedFeedback != 0}
                >
                    <FontAwesomeIcon
                        icon={faThumbsDown}
                        style={{ paddingLeft: 5 }}
                        size="4x"
                        color={selectedFeedback == -1 ? "red" : "gray"}
                    />
                </button>
            </div>}
            <div className="flex justify-center p-5">
                <Switch check={humanMode} onChange={toggleHuman} label="Human feedback" />
            </div>
            <div className="flex flex-row gap-1">
                <Button className="flex-1" onClick={togglePause}>
                    {paused ? "Resume" : "Pause"}
                </Button>
                <FpsCounter className="flex-none w-fit" value={currentFps} />
            </div>
        </div>
    );

};
