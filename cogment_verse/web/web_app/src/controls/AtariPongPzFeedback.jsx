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

import { useCallback, useState } from "react";
import { cogment_verse } from "../data_pb";
import { useDocumentKeypressListener, usePressedKeys } from "../hooks/usePressedKeys";
import { useRealTimeUpdateV2 } from "../hooks/useRealTimeUpdate_v2";
import { createLookup } from "../utils/controlLookup";
import { TEACHER_ACTOR_CLASS, TEACHER_NOOP_ACTION } from "../utils/constants";
import { Button } from "../components/Button";
import { FpsCounter } from "../components/FpsCounter";
import { KeyboardControlList } from "../components/KeyboardControlList";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faThumbsUp, faThumbsDown } from "@fortawesome/free-solid-svg-icons";

// cf. https://www.gymlibrary.ml/environments/atari/#action-space
const ATARI_LOOKUP = createLookup();
ATARI_LOOKUP.setAction(["LIKE"], new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 0 }] } }));
ATARI_LOOKUP.setAction(["DISLIKE"], new cogment_verse.PlayerAction({ value: { properties: [{ discrete: 1 }] } }));



export const AtariPongPzHfbEnvironments = ["environments.pettingzoo_atari_adapter_v2.HumanFeedbackEnvironment/pettingzoo.atari.pong_v3"];
export const AtariPongPzFeedback = ({ sendAction, fps = 30, actorClass, ...props }) => {
    const [paused, setPaused] = useState(false);
    const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
    useDocumentKeypressListener("p", togglePause);

    // const pressedKeys = usePressedKeys();

    // const computeAndSendAction = useCallback(
    //     (dt) => {
    //         if (pressedKeys.size === 0 && actorClass === TEACHER_ACTOR_CLASS) {
    //             sendAction(TEACHER_NOOP_ACTION);
    //             return;
    //         }

    //         

    //         if (pressedKeys.has("ArrowLeft")) {
    //             controls.push("LEFT");
    //         } else if (pressedKeys.has("ArrowRight")) {
    //             controls.push("RIGHT");
    //         }
    //         if (pressedKeys.has("ArrowDown")) {
    //             controls.push("DOWN");
    //         } else if (pressedKeys.has("ArrowUp")) {
    //             controls.push("UP");
    //         }
    //         if (pressedKeys.has(" ")) {
    //             controls.push("FIRE");
    //         }
    //         const action = ATARI_LOOKUP.getAction(controls);
    //         sendAction(action);
    //     },
    //     [pressedKeys, sendAction, actorClass]
    // );

    const { currentFps } = useRealTimeUpdateV2(fps, paused);
    const sendFeedback = (feedback) => {
        const controls = [];
        controls.push(feedback)
        const action = ATARI_LOOKUP.getAction(controls);
        sendAction(action);
    };

    return (
        <div {...props}>
            <div className="text-center" style={{
                paddingBottom: 10,
                padingTop: 10, 
            }}>
                <button onClick={sendFeedback("LIKE")}>
                    <FontAwesomeIcon
                        icon={faThumbsUp}
                        style={{ paddingRight: 50 }}
                        size="4x"
                    />
                </button>
                <button onClick={sendFeedback("DISLIKE")}>
                    <FontAwesomeIcon
                        icon={faThumbsDown}
                        style={{ paddingLeft: 5 }}
                        size="4x"
                    />
                </button>
            </div>
            <div className="flex flex-row gap-1">
                <Button className="flex-1" onClick={togglePause}>
                    {paused ? "Resume" : "Pause"}
                </Button>
                <FpsCounter className="flex-none w-fit" value={currentFps} />
            </div>
            <KeyboardControlList
                items={[
                    ["Left/Right/Down/Up Arrows", "Move the character"],
                    ["Space", "Jump"],
                    ["p", "Pause/Unpause"],
                ]}
            />
        </div>
    );
};
