import { useCallback, useState } from "react";
import {
  Button,
  DPAD_BUTTONS,
  useDPadPressedButtons,
  createLookup,
  FpsCounter,
  KeyboardControlList,
  serializePlayerAction,
  Space,
  TEACHER_ACTOR_CLASS,
  TEACHER_NOOP_ACTION,
  OBSERVER_ACTOR_CLASS,
  useDocumentKeypressListener,
  usePressedKeys,
  useRealTimeUpdate,
  SimplePlay,
  PlayObserver,
} from "@cogment/cogment-verse";

const ACTION_SPACE = new Space({
  discrete: {
    n: 5,
  },
});

export const SimpleTagControls = ({ sendAction, fps = 40, actorClass, observation, ...props }) => {

  const hello = 1;
  console.log("actorClass: " + actorClass)

  const [paused, setPaused] = useState(false);
  const togglePause = useCallback(() => setPaused((paused) => !paused), [setPaused]);
  useDocumentKeypressListener("p", togglePause);

  const pressedKeys = usePressedKeys();
  const { pressedButtons, isButtonPressed, setPressedButtons } = useDPadPressedButtons();
  const [activeButtons, setActiveButtons] = useState([]);

  const computeAndSendAction = useCallback(
    (dt) => {
      if (pressedKeys.has("ArrowLeft") || isButtonPressed(DPAD_BUTTONS.LEFT)) {
        setActiveButtons([DPAD_BUTTONS.LEFT]);
        sendAction(serializePlayerAction(ACTION_SPACE, 1));
        return;
      } else if (pressedKeys.has("ArrowRight") || isButtonPressed(DPAD_BUTTONS.RIGHT)) {
        setActiveButtons([DPAD_BUTTONS.RIGHT]);
        sendAction(serializePlayerAction(ACTION_SPACE, 2));
        return;
      } else if (pressedKeys.has("ArrowDown") || isButtonPressed(DPAD_BUTTONS.DOWN)) {
        setActiveButtons([DPAD_BUTTONS.DOWN]);
        sendAction(serializePlayerAction(ACTION_SPACE, 3));
        return;
      } else if (pressedKeys.has("ArrowUp") || isButtonPressed(DPAD_BUTTONS.UP)) {
        setActiveButtons([DPAD_BUTTONS.UP]);
        sendAction(serializePlayerAction(ACTION_SPACE, 4));
        return;
      }
      setActiveButtons([]);
      sendAction(serializePlayerAction(ACTION_SPACE, 0));
    },
    [isButtonPressed, pressedKeys, sendAction, setActiveButtons, isTeacher]
  );

  const { currentFps } = useRealTimeUpdate(computeAndSendAction, fps, paused);

  return (
    <div {...props}>
      <div className="flex flex-row p-5 justify-center">
        <DPad
          pressedButtons={pressedButtons}
          onPressedButtonsChange={setPressedButtons}
          activeButtons={activeButtons}
          disabled={paused}
        />
      </div>
      <div className="flex flex-row gap-1">
        <Button className="flex-1" onClick={togglePause}>
          {paused ? "Resume" : "Pause"}
        </Button>
        <FpsCounter className="flex-none w-fit" value={currentFps} />
      </div>
      <KeyboardControlList
        items={[
          ["Left/Right/Up/Down Arrows", "Move left/right/up/down"],
          ["p", "Pause/Unpause"],
        ]}
      />
    </div>
  );
};


const PlaySimpleTag = ({ actorParams, ...props }) => {
  const actorClassName = actorParams?.className;
  console.log("PlaySimpleTag | actorClassName: " + actorClassName)


  if (actorClassName === OBSERVER_ACTOR_CLASS) {
    return <PlayObserver actorParams={actorParams} {...props} />;
  }
  return <SimplePlay actorParams={actorParams} {...props} controls={SimpleTagControls} />;
};

export default PlaySimpleTag;
