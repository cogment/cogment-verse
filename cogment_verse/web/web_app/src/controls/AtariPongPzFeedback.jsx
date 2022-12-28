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
import { useDocumentKeypressListener } from "../hooks/usePressedKeys";
import { useRealTimeUpdateV2 } from "../hooks/useRealTimeUpdate_v2";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { Switch } from "../components/Switch"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faThumbsUp, faThumbsDown } from "@fortawesome/free-solid-svg-icons";

export const AtariPongPzHfbEnvironments = ["environments.pettingzoo_atari_adapter.HumanFeedbackEnvironment/pettingzoo.atari.pong_v3"];
export const AtariPongPzFeedback = ({ sendAction, fps = 30, actorClass, observation, ...props }) => {
    const [paused, setPaused] = useState(false);
    const [humanMode, setHumanMode] = useState(true);
    const [selectedFeedback, setFeedback] = useState(0);
    const [humanTurn, setHumanTurn] = useState(false);
    const [playerPos, setPlayerPos] = useState('');
    const playerName = observation?.gamePlayerName;
    const timeStep = observation?.step;
    const feedbackRequired = observation?.feedbackRequired;
    const gymAction = observation?.action
    const actionList = ['NONE', 'FIRE', 'DOWN', "UP", "DOWN", "UP"]


    useEffect(() => {
        if (playerName) {
            if (playerName.includes('first')) {
                setPlayerPos('left');
            } else {
                setPlayerPos('right');
            }
        }
    }, [playerName]);

    useEffect(() => {
        if (feedbackRequired) {
            setHumanTurn(true);
        } else {
            setHumanTurn(false);
        }
    }, [feedbackRequired]);

    const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
    const toggleHuman = useCallback(() => setHumanMode((humanMode) => !humanMode), [setHumanMode]);
    useDocumentKeypressListener("p", togglePause);
    useDocumentKeypressListener("h", toggleHuman);

    const { currentFps } = useRealTimeUpdateV2(fps, paused);

    if (!humanMode || !humanTurn) {
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
    const playerBox = 'opacity-90 py-2 rounded-full items-center text-white font-bold, px-5 text-base outline-none'
    const player1 = `bg-orange-500 ${playerBox}`;
    const player2 = `bg-green-500 ${playerBox}`;
    return (
        <div {...props}>

            <div className={playerPos == "right" ? ("flex flex-row-reverse") : ("flex flex-row")}>
                <div className={playerPos == "right" ? (player2) : (player1)}>
                    {playerName}
                </div>
                <div className={`bg-purple-500 ${playerBox}`}>
                    {actionList[gymAction]}
                </div>
            </div>
            <div className="flex p-5 flex-row justify-center items-center">
                <div className="flex justify-center p-5">
                    <Switch check={humanMode} onChange={toggleHuman} label="Human feedback" />
                </div>
                {
                    humanMode && <div className="ml-10 text-center" style={{
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
                                size="3x"
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
                                size="3x"
                                color={selectedFeedback == -1 ? "red" : "gray"}
                            />
                        </button>
                    </div>
                }
            </div>

            <div className="flex flex-row gap-1 p-3">
                <Button className="flex-1" onClick={togglePause}>
                    {paused ? "Resume" : "Pause"}
                </Button>
                <FpsCounter className="flex-none w-fit" value={currentFps} />
            </div>
        </div >
    );

};
