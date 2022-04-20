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

import { useCallback, useEffect, useState } from "react";
import { useDocumentEventListener } from "../../hooks/useDocumentEventListener";
import styles from "./Joystick.module.css";
import classNames from "classnames";

const SURFACE_SIZE = 200; // Sync with --joystick-surface-size
const STICK_SIZE = 50; // Sync with --joystick-stick-size
const TRANSLATE_MAX = SURFACE_SIZE / 2 - STICK_SIZE / 2;

const bboxClamp = (vector, lowerBound, upperBound) =>
  vector.map((coordinate, index) => Math.max(Math.min(coordinate, upperBound[index]), lowerBound[index]));

const uniformScale = (vector, scale) => vector.map((coordinate) => coordinate * scale);

export const useJoystickState = () => {
  const [joystickPosition, setJoystickPosition] = useState([0, 0]);
  const [isJoystickActive, setJoystickActive] = useState(false);
  const setJoystickState = useCallback(
    (position, active = false) => {
      setJoystickPosition(position);
      setJoystickActive(active);
    },
    [setJoystickPosition, setJoystickActive]
  );
  return { joystickPosition, isJoystickActive, setJoystickState };
};

const DEFAULT_PROPS = {
  position: [0, 0],
  active: false,
  onChange: ([x, y], active) => {},
  lowerBound: [-1, -1],
  upperBound: [1, 1],
  disabled: false,
};

export const Joystick = ({
  position = DEFAULT_PROPS.position,
  active = DEFAULT_PROPS.active,
  onChange = DEFAULT_PROPS.onChange,
  lowerBound = DEFAULT_PROPS.lowerBound,
  upperBound = DEFAULT_PROPS.upperBound,
  disabled = DEFAULT_PROPS.disabled,
  ...props
}) => {
  const [cursorInitialPosition, setCursorInitialPosition] = useState(null);
  useEffect(() => {
    if (disabled) {
      setCursorInitialPosition(null);
      onChange([0, 0], false);
    }
  }, [disabled, onChange, setCursorInitialPosition]);
  const handleMouseDown = useCallback(
    (event) => {
      if (!disabled) {
        setCursorInitialPosition([event.clientX, event.clientY]);
      }
    },
    [disabled, setCursorInitialPosition]
  );
  const handleMouseUp = useCallback(() => {
    setCursorInitialPosition(null);
    onChange([0, 0], false);
  }, [setCursorInitialPosition, onChange]);

  const handleMouseMove = useCallback(
    (event) => {
      if (cursorInitialPosition) {
        onChange(
          bboxClamp(
            uniformScale(
              [event.clientX - cursorInitialPosition[0], event.clientY - cursorInitialPosition[1]],
              1 / TRANSLATE_MAX
            ),
            lowerBound,
            upperBound
          ),
          true
        );
      }
    },
    [cursorInitialPosition, onChange, lowerBound, upperBound]
  );

  const pixelPosition = uniformScale(bboxClamp(position, lowerBound, upperBound), TRANSLATE_MAX);

  useDocumentEventListener("mousemove", handleMouseMove);
  useDocumentEventListener("mouseup", handleMouseUp);
  return (
    <div className={classNames(styles.joystick, { [styles.disabled]: disabled, [styles.active]: active })} {...props}>
      <div
        className={classNames(styles.stick)}
        onMouseDown={handleMouseDown}
        style={{
          transform: `translate(${pixelPosition[0]}px, ${pixelPosition[1]}px)`,
        }}
      ></div>
    </div>
  );
};
