import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

// One media element (the user's own recording, or a sample) drives every timeline on the page:
// waveform, spectrogram, pitch line, transcript and emotion windows all read `time` from here.
// Times are on the RAW input's timeline (what the user hears).

interface Playhead {
  time: number;
  playing: boolean;
  duration: number;
  register: (el: HTMLMediaElement | null) => void;
  seek: (t: number) => void;
  toggle: () => void;
}

const Ctx = createContext<Playhead>({
  time: 0, playing: false, duration: 0, register: () => {}, seek: () => {}, toggle: () => {},
});

export function PlayheadProvider({ children }: { children: ReactNode }) {
  const el = useRef<HTMLMediaElement | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    let frame = 0;
    const tick = () => {
      if (el.current && !el.current.paused) setTime(el.current.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  const register = useCallback((node: HTMLMediaElement | null) => {
    el.current = node;
    if (!node) return;
    node.onplay = () => setPlaying(true);
    node.onpause = () => setPlaying(false);
    node.onended = () => setPlaying(false);
    node.onloadedmetadata = () => setDuration(Number.isFinite(node.duration) ? node.duration : 0);
    node.ontimeupdate = () => setTime(node.currentTime);
  }, []);

  const seek = useCallback((t: number) => {
    if (!el.current) return;
    el.current.currentTime = Math.max(0, t);
    setTime(el.current.currentTime);
  }, []);

  const toggle = useCallback(() => {
    const node = el.current;
    if (!node) return;
    if (node.paused) void node.play();
    else node.pause();
  }, []);

  return <Ctx.Provider value={{ time, playing, duration, register, seek, toggle }}>{children}</Ctx.Provider>;
}

export const usePlayhead = () => useContext(Ctx);
