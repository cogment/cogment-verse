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

import { useEffect, useRef } from "react";
import styles from "./RenderedScreen.module.css";
import classNames from "classnames";

function bufferToBase64(buf) {
  const binstr = Array.prototype.map
    .call(buf, function (ch) {
      return String.fromCharCode(ch);
    })
    .join("");
  return btoa(binstr);
}

const DEFAULT_SCREEN_SRC = `${process.env.PUBLIC_URL}/assets/cogment-splash.png`;

export const RenderedScreen = ({ observation, overlay, className, ...props }) => {
  const canvasRef = useRef();

  useEffect(() => {
    const canvas = canvasRef?.current;
    if (!canvas) {
      return;
    }
    const pixelData = observation?.pixelData;
    if (!pixelData) {
      return;
    }

    canvas.src = "data:image/png;base64," + bufferToBase64(pixelData);
  }, [canvasRef, observation]);

  return (
    <div className={classNames(styles.container, className)} {...props}>
      <img
        className={classNames(styles.canvas, { blur: overlay != null })}
        ref={canvasRef}
        src={DEFAULT_SCREEN_SRC}
        alt="current observation rendered pixels"
      />
      {overlay ? <div className={styles.overlay}>{overlay}</div> : null}
    </div>
  );
};
