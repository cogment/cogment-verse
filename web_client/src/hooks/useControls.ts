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
  environment_implementation: string;
  action_map: KeyActionT[];
};

// https://github.com/openai/procgen/blob/c732194d6f6c929e4295c34f05fc911a1db9c7f4/procgen/env.py#L155
const procgen_combos = [
  { id: 0, name: "down left", keys: ["ArrowLeft", "ArrowDown"] },
  { id: 1, name: "left", keys: ["ArrowLeft"] },
  { id: 2, name: "up left", keys: ["ArrowLeft", "ArrowUp"] },
  { id: 3, name: "down", keys: ["ArrowDown"] },
  { id: 4, name: "nop", keys: ["KeyEscape"] },
  { id: 5, name: "up", keys: ["ArrowUp"] },
  { id: 6, name: "down right", keys: ["ArrowRight", "ArrowDown"] },
  { id: 7, name: "right", keys: ["ArrowRight"] },
  { id: 8, name: "up right", keys: ["ArrowRight", "ArrowUp"] },
  { id: 9, name: "action D", keys: ["KeyD"] },
  { id: 10, name: "action A", keys: ["KeyA"] },
  { id: 11, name: "action W", keys: ["KeyW"] },
  { id: 12, name: "action S", keys: ["KeyS"] },
  { id: 13, name: "action Q", keys: ["KeyQ"] },
  { id: 14, name: "action E", keys: ["KeyE"] },
];

const keymaps_json: KeyMapT[] = [
  {
    environment_implementation: "gym/LunarLander-v2",
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
    environment_implementation: "atari/Breakout",
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
    environment_implementation: "tetris/TetrisA-v0",
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
  {
    environment_implementation: "procgen/bigfish",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/bossfight",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/caveflyer",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/chaser",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/climber",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/coinrun",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/dodgeball",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/fruitbot",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/heist",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/jumper",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/leaper",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/maze",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/miner",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/ninja",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/plunder",
    action_map: procgen_combos,
  },
  {
    environment_implementation: "procgen/starpilot",
    action_map: procgen_combos,
  },
];

export const get_keymap = (environmentImplementation: string): KeyMapT | undefined => {
  for (const keymap of keymaps_json) {
    if (keymap.environment_implementation === environmentImplementation) {
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
