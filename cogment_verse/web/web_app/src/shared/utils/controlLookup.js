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

const NOOP = "noop";
export const createLookup = () => {
  const getCombo = (controls) => {
    if (controls.length === 0) {
      return NOOP;
    }
    return controls.sort().join("+");
  };

  const actionMap = {};

  const setAction = (controls, action) => {
    const combo = getCombo(controls);
    actionMap[combo] = action;
  };
  const getAction = (controls) => {
    const combo = getCombo(controls);
    const action = actionMap[combo];
    if (action == null) {
      throw new Error(`No action registered for ${combo}`);
    }
    return action;
  };

  return { setAction, getAction, getCombo };
};
