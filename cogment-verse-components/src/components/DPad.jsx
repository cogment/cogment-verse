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

import { useCallback, useState } from "react";
import styles from "./DPad.module.css";
import clsx from "clsx";

export const DPAD_BUTTONS = {
  UP: 0,
  DOWN: 1,
  RIGHT: 2,
  LEFT: 3,
};

const buttonArrayHas = (buttons, button) => buttons.find((buttonInArray) => buttonInArray === button) != null;

export const useDPadPressedButtons = () => {
  const [pressedButtons, setPressedButtons] = useState([]);
  const isButtonPressed = useCallback((button) => buttonArrayHas(pressedButtons, button), [pressedButtons]);
  return { pressedButtons, isButtonPressed, setPressedButtons };
};

export const DPad = ({
  pressedButtons = [],
  onPressedButtonsChange = (pressedButtons) => {},
  activeButtons = [],
  disabled = false,
  ...props
}) => {
  const isButtonPressed = useCallback((button) => buttonArrayHas(pressedButtons, button), [pressedButtons]);
  const isButtonActive = useCallback((button) => buttonArrayHas(activeButtons, button), [activeButtons]);
  const isButtonDisabled = useCallback(
    (button) => {
      if (Array.isArray(disabled)) {
        return buttonArrayHas(disabled, button);
      }
      return !!disabled;
    },
    [disabled]
  );
  const handleButtonDown = useCallback(
    (button) => {
      if (!isButtonPressed(button)) {
        onPressedButtonsChange([...pressedButtons, button]);
      }
    },
    [pressedButtons, isButtonPressed, onPressedButtonsChange]
  );

  const handleButtonUp = useCallback(
    (button) => {
      if (isButtonPressed(button)) {
        onPressedButtonsChange(pressedButtons.filter((pressedButton) => pressedButton !== button));
      }
    },
    [pressedButtons, isButtonPressed, onPressedButtonsChange]
  );

  const handleDownDown = useCallback(() => {
    handleButtonDown(DPAD_BUTTONS.DOWN);
  }, [handleButtonDown]);
  const handleDownUp = useCallback(() => {
    handleButtonUp(DPAD_BUTTONS.DOWN);
  }, [handleButtonUp]);

  const handleUpDown = useCallback(() => {
    handleButtonDown(DPAD_BUTTONS.UP);
  }, [handleButtonDown]);
  const handleUpUp = useCallback(() => {
    handleButtonUp(DPAD_BUTTONS.UP);
  }, [handleButtonUp]);

  const handleLeftDown = useCallback(() => {
    handleButtonDown(DPAD_BUTTONS.LEFT);
  }, [handleButtonDown]);
  const handleLeftUp = useCallback(() => {
    handleButtonUp(DPAD_BUTTONS.LEFT);
  }, [handleButtonUp]);

  const handleRightDown = useCallback(() => {
    handleButtonDown(DPAD_BUTTONS.RIGHT);
  }, [handleButtonDown]);
  const handleRightUp = useCallback(() => {
    handleButtonUp(DPAD_BUTTONS.RIGHT);
  }, [handleButtonUp]);

  return (
    <nav className={styles.dpad} {...props}>
      <button
        className={clsx(styles.up, {
          [styles.active]: isButtonActive(DPAD_BUTTONS.UP),
          [styles.disabled]: isButtonDisabled(DPAD_BUTTONS.UP),
        })}
        disabled={isButtonDisabled(DPAD_BUTTONS.UP)}
        onMouseDown={handleUpDown}
        onMouseUp={handleUpUp}
      >
        Up
      </button>
      <button
        className={clsx(styles.right, {
          [styles.active]: isButtonActive(DPAD_BUTTONS.RIGHT),
          [styles.disabled]: isButtonDisabled(DPAD_BUTTONS.RIGHT),
        })}
        disabled={isButtonDisabled(DPAD_BUTTONS.RIGHT)}
        onMouseDown={handleRightDown}
        onMouseUp={handleRightUp}
      >
        Right
      </button>
      <button
        className={clsx(styles.down, {
          [styles.active]: isButtonActive(DPAD_BUTTONS.DOWN),
          [styles.disabled]: isButtonDisabled(DPAD_BUTTONS.DOWN),
        })}
        disabled={isButtonDisabled(DPAD_BUTTONS.DOWN)}
        onMouseDown={handleDownDown}
        onMouseUp={handleDownUp}
      >
        Down
      </button>
      <button
        className={clsx(styles.left, {
          [styles.active]: isButtonActive(DPAD_BUTTONS.LEFT),
          [styles.disabled]: isButtonDisabled(DPAD_BUTTONS.LEFT),
        })}
        disabled={isButtonDisabled(DPAD_BUTTONS.LEFT)}
        onMouseDown={handleLeftDown}
        onMouseUp={handleLeftUp}
      >
        Left
      </button>
    </nav>
  );
};
