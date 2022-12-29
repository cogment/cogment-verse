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
import { useRealTimeUpdate } from "../hooks/useRealTimeUpdate";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";
import { createLookup } from "../utils/controlLookup";
import { TEACHER_ACTOR_CLASS, TEACHER_NOOP_ACTION } from "../utils/constants";
import { Switch } from "../components/Switch"

const ATARI_LOOKUP = createLookup();
ATARI_LOOKUP.setAction([], new cogment_verse.PlayerAction({ value: { properties: [{ simpleBox: { values: [0.0] } }] } }));
ATARI_LOOKUP.setAction(["LIKE"], new cogment_verse.PlayerAction({ value: { properties: [{ simpleBox: { values: [1.0] } }] } }));
ATARI_LOOKUP.setAction(["DISLIKE"], new cogment_verse.PlayerAction({ value: { properties: [{ simpleBox: { values: [-1.0] } }] } }));

export const AtariPongPzHfbEnvironments = ["environments.pettingzoo_atari_adapter.HumanFeedbackEnvironment/pettingzoo.atari.pong_v3"];
export const AtariPongPzFeedback = ({ sendAction, fps = 30, actorClass, observation, ...props }) => {
    const [paused, setPaused] = useState(false);
    const [humanMode, setHumanMode] = useState(true);
    const [humanTurn, setHumanTurn] = useState(false);
    const [playerPos, setPlayerPos] = useState('');
    const playerName = observation?.gamePlayerName;
    const timeStep = observation?.step;
    const feedbackRequired = observation?.feedbackRequired;
    const gymAction = observation?.action
    const actionList = ['NONE', 'FIRE', "UP", 'DOWN', "FIRE UP", " FIRE DOWN"]
    const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
    const toggleHuman = useCallback(() => setHumanMode((humanMode) => !humanMode), [setHumanMode]);
    useDocumentKeypressListener("p", togglePause);
    useDocumentKeypressListener("h", toggleHuman);

    const pressedKeys = usePressedKeys();
    useEffect(() => {
        if (playerName) {
            if (playerName.includes('first')) {
                setPlayerPos('right');
            } else {
                setPlayerPos('left');
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


    const computeAndSendAction = useCallback(
        (dt) => {
            if (pressedKeys.size === 0 && actorClass === TEACHER_ACTOR_CLASS) {
                sendAction(TEACHER_NOOP_ACTION);
                return;
            }
            const controls = [];
            let action;
            if (!humanMode || !humanTurn || timeStep % 20 == 0) {
                action = new cogment_verse.PlayerAction({ value: { properties: [{ simpleBox: { values: [0.0] } }] } });
                sendAction(action);
            } else {
                if (pressedKeys.has("l")) {
                    controls.push("LIKE");
                    action = ATARI_LOOKUP.getAction(controls);
                    sendAction(action);
                } else if (pressedKeys.has("d")) {
                    controls.push("DISLIKE");
                    action = ATARI_LOOKUP.getAction(controls);
                    sendAction(action);
                }
            }
        },
        [pressedKeys, sendAction, actorClass, humanMode, humanTurn]
    );
    const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

    const playerBox = 'opacity-90 py-2 rounded-full items-center text-white font-bold, px-5 text-base outline-none'
    const player1 = `bg-green-500 ${playerBox}`;
    const player2 = `bg-orange-500 ${playerBox}`;
    return (
        <div {...props}>
            {humanMode && (
                <div className={playerPos == "right" ? "flex flex-row-reverse" : "flex flex-row"}>
                    <div className={playerPos == "right" ? player1 : player2}>
                        {playerName}
                    </div>
                    <div className={`bg-purple-500 ${playerBox}`}>
                        {actionList[gymAction]}
                    </div>
                </div>

            )}
            <div className="flex justify-center p-5">
                <Switch check={humanMode} onChange={toggleHuman} label="Human feedback" />
            </div>
            <div className="flex flex-row gap-1 p-3">
                <Button className="flex-1" onClick={togglePause}>
                    {paused ? "Resume" : "Pause"}
                </Button>
                <FpsCounter className="flex-none w-fit" value={currentFps} />
            </div>
            <KeyboardControlList
                items={[
                    ["l", "LIKE"],
                    ["d", "DISLIKE"],
                    ["p", "Pause/Unpause"],
                    ["h", "Feedback/Not Feedback"],
                ]}
            />
        </div >
    );

};
