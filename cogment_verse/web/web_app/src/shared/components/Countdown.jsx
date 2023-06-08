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

import React, { useEffect } from "react";
import { CountdownCircleTimer } from "react-countdown-circle-timer";

export const Countdown = ({ onAfterCountdown, duration = 1000 }) => {
  useEffect(() => {
    const timeout = setTimeout(onAfterCountdown, duration);
    return () => clearTimeout(timeout);
  });

  return (
    <CountdownCircleTimer isPlaying duration={duration / 1000} colors={["#85a2d1", "#85a2d1"]} colorsTime={[1, 0]}>
      {() => <></>}
    </CountdownCircleTimer>
  );
};
