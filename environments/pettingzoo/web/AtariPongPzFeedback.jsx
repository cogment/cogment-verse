// Copyright 2023 AI Redefined Inc. <dev+cogment@ai-r.com>
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
import {
  Button,
  DType,
  FpsCounter,
  KeyboardControlList,
  serializePlayerAction,
  Space,
  Switch,
  useDocumentKeypressListener,
  useRealTimeUpdate,
  WEB_ACTOR_NAME,
} from "@cogment/cogment-verse";

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faThumbsUp, faThumbsDown, faMeh } from "@fortawesome/free-solid-svg-icons";

const ACTION_SPACE = new Space({
  box: {
    low: {
      // Only the dtype and shape of the lower bound are used to serialize the action
      dtype: DType.DTYPE_FLOAT32,
      shape: [1],
    },
  },
});

export const AtariPongPzFeedback = ({ sendAction, fps = 30, actorClass, observation, tickId, ...props }) => {
  const [paused, setPaused] = useState(false);
  const [humanMode, setHumanMode] = useState(true);
  const [selectedFeedback, setFeedback] = useState(2);
  const [humanTurn, setHumanTurn] = useState(false);
  const [playerPos, setPlayerPos] = useState("");
  const [playerNameDisplay, setPlayerNameDisplay] = useState(false);
  const playerName = observation?.gamePlayerName;
  const currentPlayer = observation?.currentPlayer;
  const gymAction = observation?.actionValue;
  const actionList = ["NONE", "FIRE", "UP", "DOWN", "FIRE UP", " FIRE DOWN"];

  useEffect(() => {
    if (playerName) {
      if (playerName.includes("first")) {
        setPlayerPos("right");
      } else {
        setPlayerPos("left");
      }
    }
  }, [playerName]);

  useEffect(() => {
    if (currentPlayer == WEB_ACTOR_NAME) {
      setHumanTurn(true);
    } else {
      setHumanTurn(false);
    }
  }, [currentPlayer]);

  useEffect(() => {
    if (!humanMode || !humanTurn || tickId % 3 != 0) {
      setPlayerNameDisplay(false);
    } else {
      setPlayerNameDisplay(true);
    }
  }, [tickId, humanMode, humanTurn]);

  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  const toggleHuman = useCallback(() => setHumanMode((humanMode) => !humanMode), [setHumanMode]);
  useDocumentKeypressListener("p", togglePause);
  useDocumentKeypressListener("h", toggleHuman);
  const { currentFps } = useRealTimeUpdate((dt) => {}, fps, paused);

  if (!humanMode || !humanTurn || tickId % 3 != 0) {
    // Neural action
    const action = serializePlayerAction(ACTION_SPACE, [0.0]);
    sendAction(action);
  }
  const sendFeedback = (feedback) => {
    let action;
    if (feedback == "LIKE") {
      setFeedback(1);
      action = serializePlayerAction(ACTION_SPACE, [1.0]);
    } else if (feedback == "DISLIKE") {
      setFeedback(-1);
      action = serializePlayerAction(ACTION_SPACE, [-1.0]);
    } else {
      setFeedback(0);
      action = serializePlayerAction(ACTION_SPACE, [0.0]);
    }
    // TODO: replace sendAction by a proper add_reward function when it is available on JS SDK
    setTimeout(() => {
      setFeedback(2);
    }, 150);
    sendAction(action);
  };
  const playerBox = "opacity-90 py-1 rounded-full items-center text-white px-2  text-sm outline-none";
  const player1 = `bg-green-500 ${playerBox}`;
  const player2 = `bg-orange-500 ${playerBox}`;
  return (
    <div {...props}>
      {playerNameDisplay ? (
        <div className={"flex p-2 flex-row justify-center items-center"}>
          <div className={playerPos == "right" ? player1 : player2}>{playerName}</div>
          <div className={`bg-purple-500 ${playerBox}`}>{actionList[gymAction]}</div>
        </div>
      ) : (
        <div></div>
      )}
      <div className="flex p-2 flex-row justify-center items-center">
        <div className="flex justify-center p-2">
          <Switch check={humanMode} onChange={toggleHuman} label="Human feedback" />
        </div>
        {humanMode && (
          <div
            className="ml-5 text-center"
            style={{
              paddingBottom: 10,
              padingTop: 10,
            }}
          >
            <button
              type="button"
              onClick={() => sendFeedback("LIKE")}
              // disabled={selectedFeedback != 0}
            >
              <FontAwesomeIcon
                icon={faThumbsUp}
                style={{ paddingRight: 5 }}
                size="2x"
                color={selectedFeedback == 1 ? "green" : "gray"}
              />
            </button>
            <button
              type="button"
              onClick={() => sendFeedback("NEURAL")}
              // disabled={selectedFeedback != 0}
            >
              <FontAwesomeIcon
                icon={faMeh}
                style={{ paddingLeft: 5 }}
                size="2x"
                color={selectedFeedback == 0 ? "blue" : "gray"}
              />
            </button>
            <button
              type="button"
              onClick={() => sendFeedback("DISLIKE")}
              // disabled={selectedFeedback != 0}
            >
              <FontAwesomeIcon
                icon={faThumbsDown}
                style={{ paddingLeft: 15 }}
                size="2x"
                color={selectedFeedback == -1 ? "red" : "gray"}
              />
            </button>
          </div>
        )}
      </div>

      <div className="flex flex-row gap-1 p-3">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
        <FpsCounter className="flex-none w-fit" value={currentFps} />
      </div>
      <KeyboardControlList
        items={[
          ["p", "Pause/Unpause"],
          ["h", "Feedback/Not Feedback"],
        ]}
      />
    </div>
  );
};
