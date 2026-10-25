import { useEffect, useRef, useState, type CSSProperties } from "react";
import { usePlayhead } from "../lib/playhead";
import type { RunState } from "../lib/run";
import type { AudioCleanData, AudioFeaturesData } from "../types";
import { Card, LANE_VAR } from "./Stage";
import { ProbBars, fmt, ordinal, seriesPath } from "./charts";

const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Raw waveform (grey), then the noise-gated one drawn over it from left to right; the trimmed
 *  silence is shaded. Everything on the RAW timeline, with the shared playhead. */
function CleaningWave({ d }: { d: AudioCleanData }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [progress, setProgress] = useState(reducedMotion() ? 1 : 0);
  const { time, seek } = usePlayhead();

  useEffect(() => {
    if (reducedMotion()) return;
    let frame = 0;
    const start = performance.now();
    const tick = () => {
      const p = Math.min(1, (performance.now() - start) / 1100);
      setProgress(p);
      if (p < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [d]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const mid = h / 2;
    const draw = (env: number[], color: string, upto = 1, scale = 1) => {
      const n = env.length;
      const bw = w / n;
      g.fillStyle = color;
      for (let i = 0; i < n * upto; i++) {
        const a = Math.min(1, env[i] * scale) * (mid - 2);
        g.fillRect(i * bw, mid - a, Math.max(1, bw - 0.4), Math.max(1, 2 * a));
      }
    };
    const peak = Math.max(...d.raw_envelope, 1e-6);
    draw(d.raw_envelope, "rgba(163,176,189,0.28)", 1, 1 / peak);
    draw(d.gated_envelope, getComputedStyle(c).getPropertyValue("--voice").trim() || "#38bdf8", progress, 1 / peak);
    // trimmed silence
    const x0 = (d.trim_start_sec / d.raw_sec) * w, x1 = (d.trim_end_sec / d.raw_sec) * w;
    g.fillStyle = "rgba(7,9,12,0.72)";
    if (progress >= 1) { g.fillRect(0, 0, x0, h); g.fillRect(x1, 0, w - x1, h); }
    // playhead
    const px = (time / d.raw_sec) * w;
    if (time > 0) { g.fillStyle = "#fff"; g.fillRect(px, 0, 1, h); }
  }, [d, progress, time]);

  return (
    <canvas ref={canvas} className="wave" style={{ height: 110, cursor: "pointer" }}
      onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); seek(((e.clientX - r.left) / r.width) * d.raw_sec); }} />
  );
}

/** Mel spectrogram with the pitch line on top and loudness under it (cleaned timeline). */
function Spectrogram({ d, trimStart }: { d: AudioFeaturesData; trimStart: number }) {
  const { time, seek } = usePlayhead();
  const W = 900, H = 200, specH = 160;
  const t = time - trimStart;
  const x = (t / d.curve_sec) * W;
  return (
    <div className="spec-wrap" onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); seek(trimStart + ((e.clientX - r.left) / r.width) * d.curve_sec); }}
      style={{ cursor: "pointer" }}>
      <img src={d.spectrogram} alt="mel spectrogram of the cleaned voice" />
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <path d={seriesPath(d.pitch, W, specH, 60, 400)} fill="none" stroke="#fff" strokeWidth="2" vectorEffect="non-scaling-stroke" opacity="0.92" />
        <rect x="0" y={specH} width={W} height={H - specH} fill="#0d1117" />
        <path d={seriesPath(d.loudness_db, W, H - specH - 6, -50, 0)} transform={`translate(0 ${specH + 3})`} fill="none"
          stroke="var(--voice)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
        {d.windows.length > 1 && d.windows.slice(1).map((w) => (
          <line key={w.start} x1={(w.start / d.curve_sec) * W} x2={(w.start / d.curve_sec) * W} y1="0" y2={H} stroke="#fff" strokeDasharray="4 4" opacity="0.5" vectorEffect="non-scaling-stroke" />
        ))}
        {t > 0 && t < d.curve_sec && <line className="playline" x1={x} x2={x} y1="0" y2={H} />}
      </svg>
    </div>
  );
}

function Measures({ d }: { d: AudioFeaturesData }) {
  const [more, setMore] = useState(false);
  const shown = d.measures.filter((m) => m.name !== "zcr_mean" && m.name !== "rms_mean");
  const unit: Record<string, (v: number) => string> = {
    pitch_mean: (v) => `${v.toFixed(0)} Hz`, pitch_std: (v) => `±${v.toFixed(0)} Hz`,
    voiced_fraction: (v) => `${(v * 100).toFixed(0)}%`, rms_std: (v) => v.toFixed(3),
    nonsilent_fraction: (v) => `${(v * 100).toFixed(0)}%`,
  };
  const lo = Math.min(...d.mfcc_means), hi = Math.max(...d.mfcc_means);
  return (
    <div>
      {shown.map((m) => (
        <div key={m.name} className="measure">
          <span>{m.label}</span>
          <span className="mono small">{unit[m.name]?.(m.value) ?? fmt(m.value, 3)} <span className="muted">· {ordinal(m.percentile)} pct</span></span>
          <span className="bar"><i style={{ width: `${m.percentile}%` }} /></span>
        </div>
      ))}
      <button className="btn ghost small" style={{ marginTop: 10, padding: "5px 10px" }} onClick={() => setMore(!more)}>
        {more ? "hide" : `+ ${d.n_features - shown.length} more numbers (40 MFCC means + spreads …)`}
      </button>
      {more && (
        <svg viewBox="0 0 400 60" style={{ width: "100%", marginTop: 8 }} aria-label="40 MFCC means">
          {d.mfcc_means.map((v, i) => {
            const h = ((v - lo) / (hi - lo || 1)) * 52 + 2;
            return <rect key={i} x={i * 10 + 1} y={58 - h} width="8" height={h} rx="1.5" fill="var(--voice)" opacity={0.35 + 0.65 * (h / 54)} />;
          })}
        </svg>
      )}
      <div className="caption">Percentile = position among the 1,500 clips of the study. The model reads all {d.n_features} numbers + 8 emotion probabilities.</div>
    </div>
  );
}

function SpeechEmotion({ d, trimStart }: { d: AudioFeaturesData; trimStart: number }) {
  const { time, playing } = usePlayhead();
  const [picked, setPicked] = useState<number | null>(null);
  const t = time - trimStart;
  const live = d.emotion_timeline.findIndex((w) => t >= w.start && t < w.end);
  const index = picked ?? (playing && live >= 0 ? live : null);
  const values = index === null ? d.emotion : d.emotion_timeline[index].probs;
  return (
    <div>
      <div className="emo-steps">
        <button className={index === null ? "on" : ""} onClick={() => setPicked(null)}>all</button>
        {d.emotion_timeline.map((w, i) => (
          <button key={i} className={index === i ? "on" : ""} onClick={() => setPicked(picked === i ? null : i)}>
            {w.start.toFixed(0)}–{w.end.toFixed(0)}s
          </button>
        ))}
      </div>
      <ProbBars labels={d.emotion_labels} values={values} />
      <div className="caption">
        {index === null
          ? `wav2vec2 on the whole ${d.windows.length > 1 ? "window" : "input"}: these 8 numbers go into the model.`
          : "One 5 s step, for display only: the model uses the whole input."} Trained on acted speech.
      </div>
    </div>
  );
}

export function VoiceLane({ state, clean, features }: { state: RunState; clean?: AudioCleanData; features?: AudioFeaturesData }) {
  return (
    <div className="lane" style={{ "--lane": LANE_VAR.voice } as CSSProperties}>
      <div className="lane-head">
        <h3><i />Voice</h3>
        <span className="small muted">the strongest lane in the study (test AUC 0.72)</span>
      </div>
      <div className="lane-grid">
        <Card lane="voice" step="V1" title="Cleaning" stage={state.stages.audio_clean} ready={!!clean} className="span-5" busyText="spectral gating, trimming silence…">
          {clean && (
            <>
              <CleaningWave d={clean} />
              <div className="chips">
                <span className="chip">background <b>{clean.noise_floor_db.toFixed(0)} dB</b></span>
                <span className="chip">gated <b>{(clean.gated_share * 100).toFixed(0)}%</b> of frames</span>
                <span className="chip">trimmed <b>{clean.removed_sec.toFixed(1)} s</b></span>
                <span className="chip">gain <b>×{clean.gain.toFixed(2)}</b></span>
              </div>
              <div className="caption">Grey: raw sound, 16 kHz mono. Blue: after spectral gating (the quietest {clean.settings.noise_quiet_percent}% of frames set the noise profile). Shaded: silence cut off.</div>
            </>
          )}
        </Card>
        <Card lane="voice" step="V2" title="Spectrogram · pitch · loudness" stage={state.stages.audio_features} ready={!!features} className="span-7" busyText="pYIN pitch tracking, 40 MFCCs…">
          {features && clean && (
            <>
              <Spectrogram d={features} trimStart={clean.trim_start_sec} />
              <div className="chips">
                <span className="chip">voiced <b>{(features.voiced_share * 100).toFixed(0)}%</b></span>
                <span className="chip">length <b>{features.duration_sec.toFixed(1)} s</b></span>
                {features.windows.length > 1 && <span className="chip"><b>{features.windows.length}</b> windows of ~15 s</span>}
              </div>
              <div className="caption">Mel spectrogram of the cleaned voice (low pitch at the bottom). White line: pitch (pYIN, 60–400 Hz); gaps are unvoiced. Blue strip: loudness. Click to seek.</div>
            </>
          )}
        </Card>
        <Card lane="voice" step="V3" title="Voice measures" stage={state.stages.audio_features} ready={!!features} className="span-6">
          {features && <Measures d={features} />}
        </Card>
        <Card lane="voice" step="V4" title="Speech emotion · wav2vec2" stage={state.stages.audio_features} ready={!!features} className="span-6" busyText="wav2vec2-large (frozen)…">
          {features && clean && <SpeechEmotion d={features} trimStart={clean.trim_start_sec} />}
        </Card>
      </div>
    </div>
  );
}
