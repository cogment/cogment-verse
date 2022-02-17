import React from "react";
import { CountdownCircleTimer } from "react-countdown-circle-timer";

interface CountdownProps {
  onAfterCountdown: () => void;
}

export const Countdown: React.FC<CountdownProps> = ({ onAfterCountdown }) => {
  setTimeout(onAfterCountdown, 1000);

  return (
    <div
      style={{
        position: "absolute",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 100,
      }}
    >
      <CountdownCircleTimer isPlaying duration={1} colors={["#85a2d1", "#85a2d1"]} colorsTime={[1, 0]}>
        {() => <></>}
      </CountdownCircleTimer>
    </div>
  );
};
