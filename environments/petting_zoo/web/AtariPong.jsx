import { OBSERVER_ACTOR_CLASS, EVALUATOR_ACTOR_CLASS, PlayObserver, SimplePlay } from "@cogment/cogment-verse";

import { AtariPongPzControls } from "./AtariPongPzControls";
import { AtariPongPzFeedback } from "./AtariPongPzFeedback";

const PlayAtariPong = ({ actorParams, ...props }) => {
  const actorClassName = actorParams?.className;

  if (actorClassName === OBSERVER_ACTOR_CLASS) {
    return <PlayObserver actorParams={actorParams} {...props} />;
  }
  if (actorClassName === EVALUATOR_ACTOR_CLASS) {
    return <SimplePlay actorParams={actorParams} {...props} controls={AtariPongPzFeedback} />;
  }
  return <SimplePlay actorParams={actorParams} {...props} controls={AtariPongPzControls} />;
};

export default PlayAtariPong;
