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

import { useEffect, useState } from "react";

export const useRealTimeUpdate = (sendAction, fps = 30, paused = true) => {
  const [currentFps, setCurrentFps] = useState(fps);
  const [lastUpdateTimestamp, setLastUpdateTimestamp] = useState(null);

  useEffect(() => {
    if (paused) {
      return;
    }

    const targetDeltaTime = 1000.0 / fps;

    const remainingDeltaTime =
      lastUpdateTimestamp != null ? Math.max(0, targetDeltaTime - new Date().getTime() + lastUpdateTimestamp) : 0;

    const timer = setTimeout(() => {
      const currentTimestamp = new Date().getTime();
      const actualDeltaTime =
        lastUpdateTimestamp != null ? new Date().getTime() - lastUpdateTimestamp : targetDeltaTime;
      sendAction(actualDeltaTime);
      setLastUpdateTimestamp(currentTimestamp);
      setCurrentFps(1000 / actualDeltaTime);
    }, remainingDeltaTime);
    return () => {
      clearTimeout(timer);
    };
  }, [paused, fps, sendAction, lastUpdateTimestamp, setLastUpdateTimestamp, setCurrentFps]);

  return { currentFps };
};
