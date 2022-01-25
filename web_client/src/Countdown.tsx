import React, { useCallback, useEffect } from 'react';

interface CountdownProps {
  onAfterCountdown: () => void;
}

export const Countdown: React.FC<CountdownProps> = ({ onAfterCountdown }) => {

  const [progress, setProgress] = React.useState(0);
  //function that calls requestAnimationFrame and sets the progress to some fraction of a second

  const updateProgress = useCallback(() => {
    setProgress(progress => progress + 0.01);
  }, []);

  useEffect(() => {
    if (progress >= 1) {
      onAfterCountdown();
    }
    else {
      setTimeout(updateProgress, 10);
    }
  }, [progress, onAfterCountdown, updateProgress]);

  return <div style={{
    position: 'absolute',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    top: "100%",
    height: '1%',
    backgroundColor: '#66f',
    width: progress * 100 + '%',
    zIndex: 1000,
  }}>
  </div>
};
