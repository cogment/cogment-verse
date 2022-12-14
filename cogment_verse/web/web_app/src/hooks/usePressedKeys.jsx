// Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
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
import { useDocumentEventListener } from "./useDocumentEventListener";

export const useDocumentKeypressListener = (key, listener) => {
  const handleKeyUp = useCallback(
    (event) => {
      if (event.key === key) {
        listener();
        event.stopPropagation();
        event.preventDefault();
      }
    },
    [key, listener]
  );
  useDocumentEventListener("keyup", handleKeyUp);
};

export const usePressedKeys = () => {
  const [pressedKeys, setPressedKeys] = useState(new Set());
  const handleKeyDown = useCallback(
    (event) => {
      event.stopPropagation();
      event.preventDefault();
      setPressedKeys((pressedKeys) => {
        pressedKeys.add(event.key);
        return pressedKeys;
      });
    },
    [setPressedKeys]
  );
  useDocumentEventListener("keydown", handleKeyDown);
  const handleKeyUp = useCallback(
    (event) => {
      event.stopPropagation();
      event.preventDefault();
      setPressedKeys((pressedKeys) => {
        pressedKeys.delete(event.key);
        return pressedKeys;
      });
    },
    [setPressedKeys]
  );
  useDocumentEventListener("keyup", handleKeyUp);
  return pressedKeys;
};
