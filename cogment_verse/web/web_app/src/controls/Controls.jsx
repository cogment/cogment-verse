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

<<<<<<< HEAD
import { useMemo } from "react";
import { EVALUATOR_ACTOR_CLASS, OBSERVER_ACTOR_CLASS, PLAYER_ACTOR_CLASS, TEACHER_ACTOR_CLASS } from "../utils/constants";
import { AtariPitfallControls, AtariPitfallEnvironments } from "./AtariPitfallControls";
import { AtariPongPzControls, AtariPongPzEnvironments } from "./AtariPongPzControls";
import { AtariPongPzFeedback, AtariPongPzHfbEnvironments } from "./AtariPongPzFeedback";
import { ConnectFourControls, ConnectFourEnvironments } from "./ConnectFourControls";
import { GymCartPoleControls, GymCartPoleEnvironments } from "./GymCartPoleControls";
import {
  GymLunarLanderContinuousControls,
  GymLunarLanderContinuousEnvironments,
} from "./GymLunarLanderContinuousControls";
import { GymLunarLanderControls, GymLunarLanderEnvironments } from "./GymLunarLanderControls";
import { GymMountainCarControls, GymMountainCarEnvironments } from "./GymMountainCarControls";
import { OvercookedRealTimeControls, OvercookedRealTimeEnvironments } from "./OvercookedRealTimeControls";
import { OvercookedTurnBasedControls, OvercookedTurnBasedEnvironments } from "./OvercookedTurnBasedControls";
import { RealTimeObserverControls } from "./RealTimeObserverControls";
import { TetrisControls, TetrisEnvironments } from "./TetrisControls";
import { TurnBasedObserverControls } from "./TurnBasedObserverControls";
=======
import { useEffect, useRef, useState } from "react";
import { Error } from "../components/Error";
import { RealTimeObserverControls } from "./RealTimeObserverControls";
import { TurnBasedObserverControls } from "./TurnBasedObserverControls";
import { ConnectFourControls, ConnectFourEnvironments } from "./ConnectFourControls";
import { AtariPitfallEnvironments, AtariPitfallControls } from "./AtariPitfallControls";
import { TetrisEnvironments, TetrisControls } from "./TetrisControls";
import { AtariPongPzEnvironments, AtariPongPzControls } from "./AtariPongPzControls";
import { AtariPongPzHfbEnvironments, AtariPongPzFeedback } from "./AtariPongPzFeedback";
import {
  WEB_BASE_URL,
  TEACHER_ACTOR_CLASS,
  PLAYER_ACTOR_CLASS,
  OBSERVER_ACTOR_CLASS,
  EVALUATOR_ACTOR_CLASS,
} from "../utils/constants";
>>>>>>> 5692c11 (WIP (+2 squashed commits))

const CONTROLS = [
  { environments: AtariPitfallEnvironments, component: AtariPitfallControls },
  { environments: TetrisEnvironments, component: TetrisControls },
  { environments: ConnectFourEnvironments, component: ConnectFourControls },
  { environments: AtariPongPzEnvironments, component: AtariPongPzControls },
  { environments: AtariPongPzHfbEnvironments, component: AtariPongPzFeedback },
  { environments: OvercookedRealTimeEnvironments, component: OvercookedRealTimeControls },
  { environments: OvercookedTurnBasedEnvironments, component: OvercookedTurnBasedControls },
];

const Loading = ({ implementation }) => <div>Loading controls for {implementation}</div>;

export const ExternalControls = (module) => (props) => {
  const controlsRootRef = useRef(null);
  const [render, setRender] = useState(null);
  useEffect(() => {
    if (controlsRootRef.current == null) {
      return;
    }
    const { mount } = module;
    const { render, unmount } = mount(controlsRootRef.current);
    setRender(render);
    return () => {
      setRender(null);
      unmount();
    };
  }, [module, controlsRootRef]);
  if (render != null) {
    render(props);
  }
  return <div ref={controlsRootRef} />;
};

export const Controls = ({
  implementation,
  actorClass,
  sendAction,
  fps,
  turnBased,
  observation,
  tickId,
  componentFile,
}) => {
  const [ControlsComponent, setControlsComponent] = useState(() => Loading);
  useEffect(() => {
    if (OBSERVER_ACTOR_CLASS === actorClass) {
      if (turnBased) {
        setControlsComponent(TurnBasedObserverControls);
      } else {
        setControlsComponent(RealTimeObserverControls);
      }
    }
    const componentUrl = `${WEB_BASE_URL}/components/environments/${implementation}/${componentFile}`;
    console.log(`importing ${componentUrl}...`);
    import(componentUrl)
      .then((componentModule) => {
        setControlsComponent(() => ExternalControls(componentModule));
      })
      .catch((error) =>
        setControlsComponent(() => ({ implementation }) => (
          <Error title={`Unable to load controls from ${implementation}`} error={error} />
        ))
      );
  }, [implementation, actorClass, turnBased, componentFile]);

  return (
    <ControlsComponent
      sendAction={sendAction}
      fps={fps}
      actorClass={actorClass}
      implementation={implementation}
      observation={observation}
      tickId={tickId}
    />
  );
};
