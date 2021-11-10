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

import { useState } from "react";

type KeyActionT = {
  id: number;
  name: string;
  keys: string[];
};

type KeyMapT = {
  env_name: string;
  action_map: KeyActionT[];
};

const keymaps_json: KeyMapT[] = [
  {
    env_name: "LunarLander-v2",
    action_map: [
      {
        id: 0,
        name: "Cut Engine",
        keys: ["ArrowUp"],
      },
      {
        id: 1,
        name: "Right Engine",
        keys: ["ArrowLeft"],
      },
      {
        id: 2,
        name: "Bottom Engine",
        keys: ["ArrowDown"],
      },
      {
        id: 3,
        name: "Left Engine",
        keys: ["ArrowRight"],
      },
    ],
  },
  {
    env_name: "Breakout",
    action_map: [
      {
        id: 0,
        name: "No Op",
        keys: ["ArrowUp"],
      },
      {
        id: 1,
        name: "Start",
        keys: ["ArrowDown"],
      },
      {
        id: 2,
        name: "Move Right",
        keys: ["ArrowRight"],
      },
      {
        id: 3,
        name: "Move Left",
        keys: ["ArrowLeft"],
      },
    ],
  },
  {
    env_name: "TetrisA-v0",
    action_map: [
      {
        id: 0,
        name: "nop",
        keys: ["KeyW"],
      },
      {
        id: 1,
        name: "A",
        keys: ["KeyL"],
      },
      {
        id: 2,
        name: "B",
        keys: ["KeyK"],
      },
      {
        id: 3,
        name: "right",
        keys: ["KeyD"],
      },
      {
        id: 4,
        name: "right + A",
        keys: ["KeyD", "KeyL"],
      },
      {
        id: 5,
        name: "right + B",
        keys: ["KeyD", "KeyK"],
      },
      {
        id: 6,
        name: "left",
        keys: ["KeyA"],
      },
      {
        id: 7,
        name: "left + A",
        keys: ["KeyA", "KeyL"],
      },
      {
        id: 8,
        name: "left + B",
        keys: ["KeyA", "KeyK"],
      },
      {
        id: 9,
        name: "down",
        keys: ["KeyS"],
      },
      {
        id: 10,
        name: "down + A",
        keys: ["KeyS", "KeyL"],
      },
      {
        id: 11,
        name: "down + B",
        keys: ["KeyS", "KeyK"],
      },
    ],
  },
];

export const get_keymap = (envType: string, envName: string): KeyMapT | undefined => {
  for (const keymap of keymaps_json) {
    if (keymap.env_name === envName) {
      return keymap;
    }
  }
  return undefined;
};

export const useControls = (): [string[], (event: any) => void, (event: any) => void] => {
  const [pressedKeys, setPressedKeys] = useState<string[]>([]);
  const onKeyDown = (event: any): void => {
    if (pressedKeys.includes(event.code)) return;

    const keys = [...pressedKeys, event.code];
    setPressedKeys(keys);
  };

  const onKeyUp = (event: any): void => {
    if (!pressedKeys.includes(event.code)) return;

    const keys = pressedKeys.filter((key) => key !== event.code);
    setPressedKeys(keys);
  };

  return [pressedKeys, onKeyDown, onKeyUp];
};
