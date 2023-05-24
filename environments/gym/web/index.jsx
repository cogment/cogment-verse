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

import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Controls } from "./GymLunarLander";
import { Play } from "../../../cogment_verse/web/web_app/src/components/Play";

const PlayPage = (props) => {
  useEffect(() => {
    setTimeout(() => props.onTrialEnd(), 5000);
  }, []);
  return <Play {...props} />;
  //return <Play {...props} controls={Controls} />;
  // return <div>{props.trialId}</div>;
};

export const mount = (container) => {
  const root = createRoot(container, { identifierPrefix: "blop" });
  const render = (props) => {
    console.log("render", props);
    root.render(
      <React.StrictMode>
        <PlayPage {...props} />
      </React.StrictMode>
    );
  };
  const unmount = () => root.unmount();
  return { render, unmount };
};
