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

import clsx from "clsx";

export const KeyboardControlList = ({ items, className, ...props }) => {
  return (
    <ul className={clsx(className, "list-disc list-inside text-sm py-3")}>
      {items
        .filter((item) => !!item)
        .map(([label, description], index) => (
          <li key={index}>
            <span className="font-semibold bg-indigo-200 lowercase">{`${label}:`}</span>
            <span className="lowercase">{` ${description}`}</span>
          </li>
        ))}
    </ul>
  );
};
