// src/components/Button.jsx
import React2 from "react";
import { Link } from "react-router-dom";
import clsx from "clsx";
var buttonClasses = [
  "font-semibold",
  "block",
  "w-full",
  "py-2",
  "px-5",
  "bg-indigo-600",
  "disabled:bg-gray-400",
  "hover:bg-indigo-900",
  "text-red",
  "disabled:text-gray-200",
  "text-center",
  "rounded"
];
var Button = ({ className, to, ...props }) => {
  if (to != null) {
    return /* @__PURE__ */ React2.createElement(Link, { to, className: clsx(className, buttonClasses), ...props });
  }
  return /* @__PURE__ */ React2.createElement("button", { className: clsx(className, buttonClasses), ...props });
};

// src/components/Countdown.jsx
import React3, { useEffect } from "react";
import { CountdownCircleTimer } from "react-countdown-circle-timer";
var Countdown = ({ onAfterCountdown, duration = 1e3 }) => {
  useEffect(() => {
    const timeout = setTimeout(onAfterCountdown, duration);
    return () => clearTimeout(timeout);
  });
  return /* @__PURE__ */ React3.createElement(CountdownCircleTimer, { isPlaying: true, duration: duration / 1e3, colors: ["#85a2d1", "#85a2d1"], colorsTime: [1, 0] }, () => /* @__PURE__ */ React3.createElement(React3.Fragment, null));
};

// src/components/DPad.jsx
import React4, { useCallback, useState } from "react";

// esbuild-css-modules-plugin-namespace:/var/folders/c3/0r3w61b97_b7hbkzggtgrr7r0000gn/T/tmp-95254-fKaGhDX6HcSi/cogment-verse-components/src/components/DPad.module.css.js
var digest = "81d34cbb2e426bc0e7ecc97d2c114d1d949bd15b8a5dca0b2734747cd6739296";
var css = `:root {
  --dpad-bg-color: #fff;
  --dpad-bg-color-hover: #eee;
  --dpad-bg-color-active: #fff;
  --dpad-bg-color-disabled: #fff;
  --dpad-fg-color: #5217b8;
  --dpad-fg-color-hover: #5217b8;
  --dpad-fg-color-active: #ffb300;
  --dpad-fg-color-disabled: #bbb;

  --dpad-button-outer-radius: 15%;
  --dpad-button-inner-radius: 50%;

  --dpad-arrow-position: 40%;
  --dpad-arrow-position-hover: 35%;
  --dpad-arrow-base: 19px;
  --dpad-arrow-height: 13px;
}

._dpad_vt3ah_20 {
  position: relative;
  display: inline-block;

  width: 200px;
  height: 200px;

  overflow: hidden;
}

/* Buttons background */

._up_vt3ah_32,
._right_vt3ah_33,
._down_vt3ah_34,
._left_vt3ah_35 {
  display: block;
  position: absolute;
  -webkit-tap-highlight-color: rgba(255, 255, 255, 0);

  line-height: 40%;
  text-align: center;
  background: var(--dpad-bg-color);
  border-color: var(--dpad-fg-color);
  border-style: solid;
  border-width: 1px;
  padding: 0px;
  color: transparent;
}

._up_vt3ah_32,
._down_vt3ah_34 {
  width: 33.3%;
  height: 43%;
}

._left_vt3ah_35,
._right_vt3ah_33 {
  width: 43%;
  height: 33%;
}

._up_vt3ah_32 {
  top: 0;
  left: 50%;
  transform: translate(-50%, 0);
  border-radius: var(--dpad-button-outer-radius) var(--dpad-button-outer-radius) var(--dpad-button-inner-radius)
    var(--dpad-button-inner-radius);
}

._down_vt3ah_34 {
  bottom: 0;
  left: 50%;
  transform: translate(-50%, 0);
  border-radius: var(--dpad-button-inner-radius) var(--dpad-button-inner-radius) var(--dpad-button-outer-radius)
    var(--dpad-button-outer-radius);
}

._left_vt3ah_35 {
  top: 50%;
  left: 0;
  transform: translate(0, -50%);
  border-radius: var(--dpad-button-outer-radius) var(--dpad-button-inner-radius) var(--dpad-button-inner-radius)
    var(--dpad-button-outer-radius);
}

._right_vt3ah_33 {
  top: 50%;
  right: 0;
  transform: translate(0, -50%);
  border-radius: 50% var(--dpad-button-outer-radius) var(--dpad-button-outer-radius) 50%;
}

/* Buttons arrows */
._up_vt3ah_32:before,
._right_vt3ah_33:before,
._down_vt3ah_34:before,
._left_vt3ah_35:before {
  content: "";
  position: absolute;
  width: 0;
  height: 0;
  border-radius: 5px;
  border-style: solid;
  transition: all 0.25s;
}

._up_vt3ah_32:before {
  top: var(--dpad-arrow-position);
  left: 50%;
  transform: translate(-50%, -50%);
  border-width: 0 var(--dpad-arrow-height) var(--dpad-arrow-base) var(--dpad-arrow-height);
  border-color: transparent transparent var(--dpad-fg-color) transparent;
}

._down_vt3ah_34:before {
  bottom: var(--dpad-arrow-position);
  left: 50%;
  transform: translate(-50%, 50%);
  border-width: var(--dpad-arrow-base) var(--dpad-arrow-height) 0px var(--dpad-arrow-height);
  border-color: var(--dpad-fg-color) transparent transparent transparent;
}

._left_vt3ah_35:before {
  left: var(--dpad-arrow-position);
  top: 50%;
  transform: translate(-50%, -50%);
  border-width: var(--dpad-arrow-height) var(--dpad-arrow-base) var(--dpad-arrow-height) 0;
  border-color: transparent var(--dpad-fg-color) transparent transparent;
}

._right_vt3ah_33:before {
  right: var(--dpad-arrow-position);
  top: 50%;
  transform: translate(50%, -50%);
  border-width: var(--dpad-arrow-height) 0 var(--dpad-arrow-height) var(--dpad-arrow-base);
  border-color: transparent transparent transparent var(--dpad-fg-color);
}

/* Hover */

._up_vt3ah_32:hover,
._right_vt3ah_33:hover,
._down_vt3ah_34:hover,
._left_vt3ah_35:hover {
  background: var(--dpad-bg-color-hover);
  border-color: var(--dpad-fg-color-hover);
}

._up_vt3ah_32:hover:before {
  top: var(--dpad-arrow-position-hover);
  border-bottom-color: var(--dpad-fg-color-hover);
}

._down_vt3ah_34:hover:before {
  bottom: var(--dpad-arrow-position-hover);
  border-top-color: var(--dpad-fg-color-hover);
}

._left_vt3ah_35:hover:before {
  left: var(--dpad-arrow-position-hover);
  border-right-color: var(--dpad-fg-color-hover);
}

._right_vt3ah_33:hover:before {
  right: var(--dpad-arrow-position-hover);
  border-left-color: var(--dpad-fg-color-hover);
}

/* Active */

._up_vt3ah_32:active,
._right_vt3ah_33:active,
._down_vt3ah_34:active,
._left_vt3ah_35:active,
._up_vt3ah_32._active_vt3ah_175,
._right_vt3ah_33._active_vt3ah_175,
._down_vt3ah_34._active_vt3ah_175,
._left_vt3ah_35._active_vt3ah_175 {
  background: var(--dpad-bg-color-active);
  border-color: var(--dpad-fg-color-active);
}

._up_vt3ah_32:active:before,
._up_vt3ah_32._active_vt3ah_175:before {
  border-bottom-color: var(--dpad-fg-color-active);
}

._down_vt3ah_34:active:before,
._down_vt3ah_34._active_vt3ah_175:before {
  border-top-color: var(--dpad-fg-color-active);
}

._left_vt3ah_35:active:before,
._left_vt3ah_35._active_vt3ah_175:before {
  border-right-color: var(--dpad-fg-color-active);
}

._right_vt3ah_33:active:before,
._right_vt3ah_33._active_vt3ah_175:before {
  border-left-color: var(--dpad-fg-color-active);
}

/* Disabled */

._up_vt3ah_32._disabled_vt3ah_205,
._right_vt3ah_33._disabled_vt3ah_205,
._down_vt3ah_34._disabled_vt3ah_205,
._left_vt3ah_35._disabled_vt3ah_205 {
  background: var(--dpad-bg-color-disabled);
  border-color: var(--dpad-fg-color-disabled);
}

._up_vt3ah_32._disabled_vt3ah_205:before {
  top: var(--dpad-arrow-position);
  border-bottom-color: var(--dpad-fg-color-disabled);
}

._down_vt3ah_34._disabled_vt3ah_205:before {
  bottom: var(--dpad-arrow-position);
  border-top-color: var(--dpad-fg-color-disabled);
}

._left_vt3ah_35._disabled_vt3ah_205:before {
  left: var(--dpad-arrow-position);
  border-right-color: var(--dpad-fg-color-disabled);
}

._right_vt3ah_33._disabled_vt3ah_205:before {
  right: var(--dpad-arrow-position);
  border-left-color: var(--dpad-fg-color-disabled);
}
`;
(function() {
  if (typeof document === "undefined") {
    return;
  }
  if (!document.getElementById(digest)) {
    var el = document.createElement("style");
    el.id = digest;
    el.textContent = css;
    document.head.appendChild(el);
  }
})();
var DPad_module_css_default = { "dpad": "_dpad_vt3ah_20", "up": "_up_vt3ah_32", "right": "_right_vt3ah_33", "down": "_down_vt3ah_34", "left": "_left_vt3ah_35", "active": "_active_vt3ah_175", "disabled": "_disabled_vt3ah_205" };

// src/components/DPad.jsx
import clsx2 from "clsx";
var DPAD_BUTTONS = {
  UP: 0,
  DOWN: 1,
  RIGHT: 2,
  LEFT: 3
};
var buttonArrayHas = (buttons, button) => buttons.find((buttonInArray) => buttonInArray === button) != null;
var useDPadPressedButtons = () => {
  const [pressedButtons, setPressedButtons] = useState([]);
  const isButtonPressed = useCallback((button) => buttonArrayHas(pressedButtons, button), [pressedButtons]);
  return { pressedButtons, isButtonPressed, setPressedButtons };
};
var DPad = ({
  pressedButtons = [],
  onPressedButtonsChange = (pressedButtons2) => {
  },
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
  return /* @__PURE__ */ React4.createElement("nav", { className: DPad_module_css_default.dpad, ...props }, /* @__PURE__ */ React4.createElement(
    "button",
    {
      className: clsx2(DPad_module_css_default.up, {
        [DPad_module_css_default.active]: isButtonActive(DPAD_BUTTONS.UP),
        [DPad_module_css_default.disabled]: isButtonDisabled(DPAD_BUTTONS.UP)
      }),
      disabled: isButtonDisabled(DPAD_BUTTONS.UP),
      onMouseDown: handleUpDown,
      onMouseUp: handleUpUp
    },
    "Up"
  ), /* @__PURE__ */ React4.createElement(
    "button",
    {
      className: clsx2(DPad_module_css_default.right, {
        [DPad_module_css_default.active]: isButtonActive(DPAD_BUTTONS.RIGHT),
        [DPad_module_css_default.disabled]: isButtonDisabled(DPAD_BUTTONS.RIGHT)
      }),
      disabled: isButtonDisabled(DPAD_BUTTONS.RIGHT),
      onMouseDown: handleRightDown,
      onMouseUp: handleRightUp
    },
    "Right"
  ), /* @__PURE__ */ React4.createElement(
    "button",
    {
      className: clsx2(DPad_module_css_default.down, {
        [DPad_module_css_default.active]: isButtonActive(DPAD_BUTTONS.DOWN),
        [DPad_module_css_default.disabled]: isButtonDisabled(DPAD_BUTTONS.DOWN)
      }),
      disabled: isButtonDisabled(DPAD_BUTTONS.DOWN),
      onMouseDown: handleDownDown,
      onMouseUp: handleDownUp
    },
    "Down"
  ), /* @__PURE__ */ React4.createElement(
    "button",
    {
      className: clsx2(DPad_module_css_default.left, {
        [DPad_module_css_default.active]: isButtonActive(DPAD_BUTTONS.LEFT),
        [DPad_module_css_default.disabled]: isButtonDisabled(DPAD_BUTTONS.LEFT)
      }),
      disabled: isButtonDisabled(DPAD_BUTTONS.LEFT),
      onMouseDown: handleLeftDown,
      onMouseUp: handleLeftUp
    },
    "Left"
  ));
};

// src/components/FpsCounter.jsx
import React5 from "react";
import clsx3 from "clsx";
var FpsCounter = ({ value, className, ...props }) => {
  return /* @__PURE__ */ React5.createElement(
    "div",
    {
      className: clsx3(
        className,
        "text-sm",
        "py-2",
        "px-5",
        "bg-slate-600",
        "text-white",
        "text-center",
        "rounded-full"
      ),
      ...props
    },
    value.toFixed(0).padStart(2, "0"),
    " fps"
  );
};

// src/components/Joystick.jsx
import React6, { useCallback as useCallback2, useEffect as useEffect3, useState as useState2 } from "react";

// src/hooks/useDocumentEventListener.js
import { useEffect as useEffect2 } from "react";
var useDocumentEventListener = (eventName, listener) => {
  useEffect2(() => {
    document.addEventListener(eventName, listener);
    return () => {
      document.removeEventListener(eventName, listener);
    };
  }, [eventName, listener]);
};

// esbuild-css-modules-plugin-namespace:/var/folders/c3/0r3w61b97_b7hbkzggtgrr7r0000gn/T/tmp-95254-gWVwS6InBjNZ/cogment-verse-components/src/components/Joystick.module.css.js
var digest2 = "932fb0ee753c1e7aafb1b05f3666a6102b14d9efca0362add5a2d7ef3d826409";
var css2 = `:root {
  --joystick-surface-size: 200px;
  --joystick-stick-size: 50px;

  --joystick-color: #5217b8;
  --joystick-color-active: #ffb300;
  --joystick-color-disabled: #bbb;
}

._joystick_1jpad_10 {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;

  overflow: hidden;

  height: var(--joystick-surface-size);
  width: var(--joystick-surface-size);
  border-radius: calc(var(--joystick-stick-size) / 2);
  border-color: var(--joystick-color);
  border-width: 1px;
  border-style: solid;

  transition: all 0.25s;
}

._joystick_1jpad_10 > ._stick_1jpad_28 {
  height: var(--joystick-stick-size);
  width: var(--joystick-stick-size);

  border-radius: 50%;
  background-color: var(--joystick-color);

  transition: all 0.25s;
}

._joystick_1jpad_10._active_1jpad_38 {
  border-color: var(--joystick-color-active);
}

._joystick_1jpad_10._active_1jpad_38 > ._stick_1jpad_28 {
  background-color: var(--joystick-color-active);
}

._joystick_1jpad_10._disabled_1jpad_46 {
  border-color: var(--joystick-color-disabled);
}

._joystick_1jpad_10._disabled_1jpad_46 > ._stick_1jpad_28 {
  background-color: var(--joystick-color-disabled);
}
`;
(function() {
  if (typeof document === "undefined") {
    return;
  }
  if (!document.getElementById(digest2)) {
    var el = document.createElement("style");
    el.id = digest2;
    el.textContent = css2;
    document.head.appendChild(el);
  }
})();
var Joystick_module_css_default = { "joystick": "_joystick_1jpad_10", "stick": "_stick_1jpad_28", "active": "_active_1jpad_38", "disabled": "_disabled_1jpad_46" };

// src/components/Joystick.jsx
import clsx4 from "clsx";
var SURFACE_SIZE = 200;
var STICK_SIZE = 50;
var TRANSLATE_MAX = SURFACE_SIZE / 2 - STICK_SIZE / 2;
var bboxClamp = (vector, lowerBound, upperBound) => vector.map((coordinate, index) => Math.max(Math.min(coordinate, upperBound[index]), lowerBound[index]));
var uniformScale = (vector, scale) => vector.map((coordinate) => coordinate * scale);
var useJoystickState = () => {
  const [joystickPosition, setJoystickPosition] = useState2([0, 0]);
  const [isJoystickActive, setJoystickActive] = useState2(false);
  const setJoystickState = useCallback2(
    (position, active = false) => {
      setJoystickPosition(position);
      setJoystickActive(active);
    },
    [setJoystickPosition, setJoystickActive]
  );
  return { joystickPosition, isJoystickActive, setJoystickState };
};
var DEFAULT_PROPS = {
  position: [0, 0],
  active: false,
  onChange: ([x, y], active) => {
  },
  lowerBound: [-1, -1],
  upperBound: [1, 1],
  disabled: false
};
var Joystick = ({
  position = DEFAULT_PROPS.position,
  active = DEFAULT_PROPS.active,
  onChange = DEFAULT_PROPS.onChange,
  lowerBound = DEFAULT_PROPS.lowerBound,
  upperBound = DEFAULT_PROPS.upperBound,
  disabled = DEFAULT_PROPS.disabled,
  ...props
}) => {
  const [cursorInitialPosition, setCursorInitialPosition] = useState2(null);
  useEffect3(() => {
    if (disabled) {
      setCursorInitialPosition(null);
      onChange([0, 0], false);
    }
  }, [disabled, onChange, setCursorInitialPosition]);
  const handleMouseDown = useCallback2(
    (event) => {
      if (!disabled) {
        setCursorInitialPosition([event.clientX, event.clientY]);
      }
    },
    [disabled, setCursorInitialPosition]
  );
  const handleMouseUp = useCallback2(() => {
    setCursorInitialPosition(null);
    onChange([0, 0], false);
  }, [setCursorInitialPosition, onChange]);
  const handleMouseMove = useCallback2(
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
  return /* @__PURE__ */ React6.createElement("div", { className: clsx4(Joystick_module_css_default.joystick, { [Joystick_module_css_default.disabled]: disabled, [Joystick_module_css_default.active]: active }), ...props }, /* @__PURE__ */ React6.createElement(
    "div",
    {
      className: clsx4(Joystick_module_css_default.stick),
      onMouseDown: handleMouseDown,
      style: {
        transform: `translate(${pixelPosition[0]}px, ${pixelPosition[1]}px)`
      }
    }
  ));
};

// src/components/KeyboardControlList.jsx
import clsx5 from "clsx";
var KeyboardControlList = ({ items, className, ...props }) => {
  return /* @__PURE__ */ React.createElement("ul", { className: clsx5(className, "list-disc list-inside text-sm py-3") }, items.filter((item) => !!item).map(([label, description], index) => /* @__PURE__ */ React.createElement("li", { key: index }, /* @__PURE__ */ React.createElement("span", { className: "font-semibold bg-indigo-200 lowercase" }, `${label}:`), /* @__PURE__ */ React.createElement("span", { className: "lowercase" }, ` ${description}`))));
};

// src/components/Link.jsx
import React7 from "react";
import clsx6 from "clsx";
import { Link as UnstyledLink } from "react-router-dom";
var Link2 = ({ className, ...props }) => {
  return /* @__PURE__ */ React7.createElement(UnstyledLink, { className: clsx6(className, buttonClasses), ...props });
};

// src/components/RenderedScreen.jsx
import React8, { useEffect as useEffect4, useRef } from "react";

// esbuild-css-modules-plugin-namespace:/var/folders/c3/0r3w61b97_b7hbkzggtgrr7r0000gn/T/tmp-95254-53999eTdMvQH/cogment-verse-components/src/components/RenderedScreen.module.css.js
var digest3 = "c72ab0df7db0ef36606b44fb3937c584ccf2c7cdddcd65ad172e0ddfd1364c69";
var css3 = `._container_wgdhb_1 {
  position: relative;
  display: grid;
  justify-content: center;
  height: 100%;
}

._canvas_wgdhb_8,
._overlay_wgdhb_9 {
  height: 75vh;
  grid-area: 1/-1;
  z-index: 0;

  width: auto;
}

._overlay_wgdhb_9 {
  z-index: 1;

  display: flex;
  justify-content: center;
  align-items: center;
}
`;
(function() {
  if (typeof document === "undefined") {
    return;
  }
  if (!document.getElementById(digest3)) {
    var el = document.createElement("style");
    el.id = digest3;
    el.textContent = css3;
    document.head.appendChild(el);
  }
})();
var RenderedScreen_module_css_default = { "container": "_container_wgdhb_1", "canvas": "_canvas_wgdhb_8", "overlay": "_overlay_wgdhb_9" };

// src/components/RenderedScreen.jsx
import clsx7 from "clsx";
function bufferToBase64(buf) {
  const binstr = Array.prototype.map.call(buf, function(ch) {
    return String.fromCharCode(ch);
  }).join("");
  return btoa(binstr);
}
var DEFAULT_SCREEN_SRC = `${process.env.PUBLIC_URL}/assets/cogment-splash.png`;
var RenderedScreen = ({ observation, overlay, className, ...props }) => {
  const canvasRef = useRef();
  const teacherOverride = observation?.overriddenPlayers != null && observation.overriddenPlayers.length > 0;
  useEffect4(() => {
    const canvas = canvasRef?.current;
    if (!canvas) {
      return;
    }
    const renderedFrame = observation?.renderedFrame;
    if (!renderedFrame) {
      return;
    }
    canvas.src = "data:image/png;base64," + bufferToBase64(renderedFrame);
  }, [canvasRef, observation]);
  return /* @__PURE__ */ React8.createElement("div", { className: clsx7(RenderedScreen_module_css_default.container, className), ...props }, /* @__PURE__ */ React8.createElement(
    "img",
    {
      className: clsx7(RenderedScreen_module_css_default.canvas, { blur: overlay != null }),
      ref: canvasRef,
      src: DEFAULT_SCREEN_SRC,
      alt: "current observation rendered pixels"
    }
  ), overlay ? /* @__PURE__ */ React8.createElement("div", { className: RenderedScreen_module_css_default.overlay }, overlay) : null, teacherOverride ? /* @__PURE__ */ React8.createElement("div", { className: clsx7(RenderedScreen_module_css_default.overlay, "ring-inset", "ring-8", "ring-sky-500:80", "duration-75") }) : null);
};

// esbuild-css-modules-plugin-namespace:/var/folders/c3/0r3w61b97_b7hbkzggtgrr7r0000gn/T/tmp-95254-WhuYlnc451bV/cogment-verse-components/src/components/Switch.module.css.js
var digest4 = "5ffbb14ebd24f55b260bffe1d8f9f605ad7e856b5b329929b118f9c697bcd76a";
var css4 = `/* The switch - the box around the slider */
._switch_1qoun_2 {
    position: relative;
    display: inline-block;
    width: 60px;
    height: 34px;
  }
  
  /* Hide default HTML checkbox */
  ._switch_1qoun_2 input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  
  /* The slider */
  ._slider_1qoun_17 {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    -webkit-transition: .4s;
    transition: .4s;
  }
  
  ._slider_1qoun_17:before {
    position: absolute;
    content: "";
    height: 26px;
    width: 26px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    -webkit-transition: .4s;
    transition: .4s;
  }
  
  input:checked + ._slider_1qoun_17 {
    background-color: #2196F3;
  }
  
  input:focus + ._slider_1qoun_17 {
    box-shadow: 0 0 1px #2196F3;
  }
  
  input:checked + ._slider_1qoun_17:before {
    -webkit-transform: translateX(26px);
    -ms-transform: translateX(26px);
    transform: translateX(26px);
  }
  
  /* Rounded sliders */
  ._slider_1qoun_17._round_1qoun_56 {
    border-radius: 34px;
  }
  
  ._slider_1qoun_17._round_1qoun_56:before {
    border-radius: 50%;
  }`;
(function() {
  if (typeof document === "undefined") {
    return;
  }
  if (!document.getElementById(digest4)) {
    var el = document.createElement("style");
    el.id = digest4;
    el.textContent = css4;
    document.head.appendChild(el);
  }
})();
var Switch_module_css_default = { "switch": "_switch_1qoun_2", "slider": "_slider_1qoun_17", "round": "_round_1qoun_56" };

// src/components/Switch.jsx
import clsx8 from "clsx";
var Switch = ({ check, onChange, label }) => {
  return /* @__PURE__ */ React.createElement("div", { className: "flex items-center gap-2" }, /* @__PURE__ */ React.createElement("span", null, label), /* @__PURE__ */ React.createElement("label", { className: Switch_module_css_default.switch }, /* @__PURE__ */ React.createElement("input", { type: "checkbox", checked: check, onChange }), /* @__PURE__ */ React.createElement("span", { className: clsx8(Switch_module_css_default.slider, Switch_module_css_default.round) })));
};

// src/hooks/useJoinedTrial.js
import { Context } from "@cogment/cogment-js-sdk";
import React9, { useEffect as useEffect5, useRef as useRef2, useState as useState3 } from "react";

// src/constants.js
var WEB_ACTOR_NAME = "web_actor";
var TEACHER_ACTOR_CLASS = "teacher";
var PLAYER_ACTOR_CLASS = "player";
var OBSERVER_ACTOR_CLASS = "observer";
var EVALUATOR_ACTOR_CLASS = "evaluator";

// src/hooks/useJoinedTrial.js
var TRIAL_STATUS = {
  JOINING: "JOINING",
  ONGOING: "ONGOING",
  ENDED: "ENDED",
  ERROR: "ERROR"
};
var useJoinedTrial = (cogSettings, cogmentOrchestratorWebEndpoint, trialId, timeout = 5e3) => {
  console.log("in useJoinedTrial - React=", React9);
  console.log("in useJoinedTrial - useState=", useState3);
  const [[status, error], setTrialStatus] = useState3([TRIAL_STATUS.JOINING, null]);
  const [event, setEvent] = useState3({
    observation: void 0,
    message: void 0,
    reward: void 0,
    last: false,
    tickId: 0
  });
  const [sendAction, setSendAction] = useState3();
  const [actorParams, setActorParams] = useState3(null);
  const actionLock = useRef2(false);
  useEffect5(() => {
    setTrialStatus([TRIAL_STATUS.JOINING, null]);
    const context = new Context(cogSettings, "cogment_verse_web");
    const timeoutId = setTimeout(() => {
      const error2 = new Error("Joined trial didn't start actor after timeout");
      console.error(`Error while running trial [${trialId}]`, error2);
      setTrialStatus([TRIAL_STATUS.ERROR, error2]);
    }, timeout);
    const actorImplementation = async (actorSession) => {
      try {
        if (actorSession.getTrialId() !== trialId) {
          throw new Error(
            `Unexpected error, joined trial [${actorSession.getTrialId()}] doesn't match desired trial [${trialId}]`
          );
        }
        setActorParams({
          name: actorSession.name,
          // WEB_ACTOR_NAME
          config: actorSession.config,
          className: actorSession.className
        });
        setSendAction(() => (action) => {
          if (actionLock.current) {
            console.warn(
              `trial [${actorSession.getTrialId()}] at tick [${actorSession.getTickId()}] received a 2nd action, ignoring it.`
            );
            return;
          }
          actorSession.doAction(action);
          actionLock.current = true;
        });
        actionLock.current = false;
        actorSession.start();
        clearTimeout(timeoutId);
        setTrialStatus([TRIAL_STATUS.ONGOING, null]);
        let tickId = actorSession.getTickId();
        for await (const { observation, messages, rewards, type } of actorSession.eventLoop()) {
          let nextEvent = {
            observation,
            message: messages[0],
            reward: rewards[0],
            last: type === 3,
            tickId: actorSession.getTickId()
          };
          const newTick = nextEvent.tickId !== tickId;
          setEvent(nextEvent);
          if (newTick) {
            actionLock.current = false;
          }
          tickId = nextEvent.tickId;
        }
        setTrialStatus([TRIAL_STATUS.ENDED, null]);
      } catch (error2) {
        setTrialStatus([TRIAL_STATUS.ERROR, error2]);
        console.error(`Error while running trial [${trialId}]`, error2);
        throw error2;
      }
    };
    context.registerActor(
      actorImplementation,
      WEB_ACTOR_NAME,
      PLAYER_ACTOR_CLASS
      // actually what we should do is [TEACHER_ACTOR_CLASS, PLAYER_ACTOR_CLASS, OBSERVER_ACTOR_CLASS]
    );
    context.joinTrial(trialId, cogmentOrchestratorWebEndpoint, WEB_ACTOR_NAME).catch((error2) => {
      setTrialStatus([TRIAL_STATUS.ERROR, error2]);
      console.error(`Error while running trial [${trialId}]`, error2);
    });
    return () => clearTimeout(timeoutId);
  }, [cogSettings, cogmentOrchestratorWebEndpoint, trialId, timeout]);
  return [status, actorParams, event, sendAction, error];
};

// src/hooks/usePressedKeys.jsx
import React10, { useCallback as useCallback3, useState as useState4 } from "react";
var useDocumentKeypressListener = (key, listener) => {
  const handleKeyUp = useCallback3(
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
var usePressedKeys = () => {
  const [pressedKeys, setPressedKeys] = useState4(/* @__PURE__ */ new Set());
  const handleKeyDown = useCallback3(
    (event) => {
      event.stopPropagation();
      event.preventDefault();
      setPressedKeys((pressedKeys2) => {
        pressedKeys2.add(event.key);
        return pressedKeys2;
      });
    },
    [setPressedKeys]
  );
  useDocumentEventListener("keydown", handleKeyDown);
  const handleKeyUp = useCallback3(
    (event) => {
      event.stopPropagation();
      event.preventDefault();
      setPressedKeys((pressedKeys2) => {
        pressedKeys2.delete(event.key);
        return pressedKeys2;
      });
    },
    [setPressedKeys]
  );
  useDocumentEventListener("keyup", handleKeyUp);
  return pressedKeys;
};

// src/hooks/useRealTimeUpdate.jsx
import React11, { useEffect as useEffect6, useState as useState5 } from "react";
var useRealTimeUpdate = (sendAction, fps = 30, paused = true) => {
  const [currentFps, setCurrentFps] = useState5(fps);
  const [lastUpdateTimestamp, setLastUpdateTimestamp] = useState5(null);
  useEffect6(() => {
    if (paused) {
      return;
    }
    const targetDeltaTime = 1e3 / fps;
    const remainingDeltaTime = lastUpdateTimestamp != null ? Math.max(0, targetDeltaTime - (/* @__PURE__ */ new Date()).getTime() + lastUpdateTimestamp) : 0;
    const timer = setTimeout(() => {
      const currentTimestamp = (/* @__PURE__ */ new Date()).getTime();
      const actualDeltaTime = lastUpdateTimestamp != null ? (/* @__PURE__ */ new Date()).getTime() - lastUpdateTimestamp : targetDeltaTime;
      sendAction(actualDeltaTime);
      setLastUpdateTimestamp(currentTimestamp);
      setCurrentFps(1e3 / actualDeltaTime);
    }, remainingDeltaTime);
    return () => {
      clearTimeout(timer);
    };
  }, [paused, fps, sendAction, lastUpdateTimestamp, setLastUpdateTimestamp, setCurrentFps]);
  return { currentFps };
};
export {
  Button,
  Countdown,
  DPAD_BUTTONS,
  DPad,
  EVALUATOR_ACTOR_CLASS,
  FpsCounter,
  Joystick,
  KeyboardControlList,
  Link2 as Link,
  OBSERVER_ACTOR_CLASS,
  PLAYER_ACTOR_CLASS,
  RenderedScreen,
  Switch,
  TEACHER_ACTOR_CLASS,
  TRIAL_STATUS,
  WEB_ACTOR_NAME,
  useDPadPressedButtons,
  useDocumentEventListener,
  useDocumentKeypressListener,
  useJoinedTrial,
  useJoystickState,
  usePressedKeys,
  useRealTimeUpdate
};
//# sourceMappingURL=index.js.map
