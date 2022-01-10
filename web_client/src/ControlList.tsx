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

import { get_keymap } from "./hooks/useControls";

export const ControlList = (props: any) => {
  // const keymaps = keymaps;
  const { environmentImplementation } = props;
  const keymap = get_keymap(environmentImplementation);

  if (keymap === undefined) {
    return (
      <div>
        {/* No control scheme available for environment `{environmentImplementation}` */}
        No trial in progress
      </div>
    );
  }

  //console.log(keymap.action_map);

  return (
    <div>
      <table>
        <thead>
          <tr>
            <th>{keymap.environment_implementation} Controls</th>
          </tr>
          <tr>
            <th>Action</th>
            <th>Action Name</th>
            <th>Key Combination</th>
          </tr>
        </thead>
        <tbody>
          {keymap.action_map.map((action, i) => {
            return (
              <tr key={action.id}>
                <td>{action.id}</td>
                <td>{action.name}</td>
                <td>{action.keys.join(" + ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};
