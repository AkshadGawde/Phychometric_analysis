import { useEffect, useRef, useState } from "react";
import { getSamples } from "../lib/api";
import type { Sample } from "../types";
import { clock } from "./charts";

export type RunInput =
  | { kind: "live"; blob: Blob; filename: string; mediaUrl: string; isVideo: boolean; title: string }
  | { kind: "sample"; sample: Sample };

type Tab = "voice" | "video" | "upload" | "samples";

const VOICE_MIN_SEC = 5;
const VOICE_MAX_SEC = 30;
const VIDEO_SEC = 15;            // as long as the training clips

function pickMime(kind: "audio" | "video") {
  const options = kind === "audio"
    ? ["audio/webm;codecs=opus", "audio/mp4", "audio/webm", "audio/ogg"]
    : ["video/mp4", "video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
  return options.find((m) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) ?? "";
}
const extFor = (mime: string) => (mime.includes("mp4") ? ".mp4" : mime.includes("ogg") ? ".ogg" : ".webm");

/** Live microphone level from an AnalyserNode: the meter shows the real input. */
function useLevel(stream: MediaStream | null, bands = 18) {
  const [levels, setLevels] = useState<number[]>(() => Array(bands).fill(0));
  useEffect(() => {
    if (!stream || stream.getAudioTracks().length === 0) return;
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const bins = new Uint8Array(analyser.frequencyBinCount);
    let frame = 0;
    const tick = () => {
      analyser.getByteFrequencyData(bins);
      const per = Math.floor(bins.length / 2 / bands);         // speech lives in the lower half
      setLevels(Array.from({ length: bands }, (_, b) => {
        let s = 0;
        for (let i = 0; i < per; i++) s += bins[b * per + i];
        return s / per / 255;
      }));
      frame = requestAnimationFrame(tick);
    };
    tick();
    return () => { cancelAnimationFrame(frame); void ctx.close(); };
  }, [stream, bands]);
  return levels;
}

function Meter({ levels }: { levels: number[] }) {
  return <div className="meter" aria-hidden>{levels.map((l, i) => <i key={i} style={{ height: `${6 + l * 40}px` }} />)}</div>;
}

function useRecorder(kind: "audio" | "video") {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<{ blob: Blob; url: string; mime: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const started = useRef(0);

  const open = async () => {
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia(kind === "audio"
        ? { audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true } }
        : { audio: { echoCancellation: false, noiseSuppression: false }, video: { width: 640, height: 480, facingMode: "user" } });
      setStream(s);
      return s;
    } catch (e) {
      setError(`${kind === "audio" ? "Microphone" : "Camera"} not available: ${(e as Error).message}`);
      return null;
    }
  };

  const start = (s: MediaStream) => {
    const mime = pickMime(kind);
    const rec = new MediaRecorder(s, mime ? { mimeType: mime } : undefined);
    chunks.current = [];
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    rec.onstop = () => {
      const type = rec.mimeType || mime || (kind === "audio" ? "audio/webm" : "video/webm");
      const blob = new Blob(chunks.current, { type });
      setResult({ blob, url: URL.createObjectURL(blob), mime: type });
      setRecording(false);
    };
    recorder.current = rec;
    rec.start(250);
    started.current = performance.now();
    setElapsed(0);
    setResult(null);
    setRecording(true);
  };

  const stop = () => recorder.current?.state === "recording" && recorder.current.stop();

  useEffect(() => {
    if (!recording) return;
    const id = setInterval(() => setElapsed((performance.now() - started.current) / 1000), 100);
    return () => clearInterval(id);
  }, [recording]);

  const close = () => { stream?.getTracks().forEach((t) => t.stop()); setStream(null); };
  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);

  return { stream, recording, elapsed, result, error, open, start, stop, close, reset: () => setResult(null) };
}

function VoiceTab({ onReady }: { onReady: (blob: Blob, filename: string, url: string, isVideo: boolean) => void }) {
  const r = useRecorder("audio");
  const levels = useLevel(r.stream);
  useEffect(() => { if (r.recording && r.elapsed >= VOICE_MAX_SEC) r.stop(); }, [r.elapsed, r.recording]);
  useEffect(() => { if (r.result) r.close(); }, [r.result]);

  const toggle = async () => {
    if (r.recording) { if (r.elapsed >= VOICE_MIN_SEC) r.stop(); return; }
    const s = r.stream ?? (await r.open());
    if (s) r.start(s);
  };

  return (
    <div className="recorder">
      <p className="ink2" style={{ margin: 0, textAlign: "center", maxWidth: 460 }}>
        Talk for {VOICE_MIN_SEC}–{VOICE_MAX_SEC} seconds about anything: your weekend, a hobby, what you had for lunch.
        About 15 s matches the training clips.
      </p>
      <Meter levels={levels} />
      <button className={`rec-button ${r.recording ? "on" : ""}`} onClick={toggle}
        disabled={r.recording && r.elapsed < VOICE_MIN_SEC} aria-label={r.recording ? "Stop recording" : "Start recording"}>
        <i />
      </button>
      <div className="mono" style={{ fontSize: 20 }}>{clock(r.elapsed)} <span className="muted small">/ {clock(VOICE_MAX_SEC)}</span></div>
      {r.recording && r.elapsed < VOICE_MIN_SEC && <div className="muted small">keep going, at least {VOICE_MIN_SEC} s</div>}
      {r.error && <div className="error-box">{r.error}</div>}
      {r.result && (
        <div style={{ display: "grid", gap: 12, justifyItems: "center" }}>
          <audio controls src={r.result.url} />
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn ghost" onClick={r.reset}>Record again</button>
            <button className="btn primary" onClick={() => onReady(r.result!.blob, `voice${extFor(r.result!.mime)}`, r.result!.url, false)}>
              Run the pipeline →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function VideoTab({ onReady }: { onReady: (blob: Blob, filename: string, url: string, isVideo: boolean) => void }) {
  const r = useRecorder("video");
  const preview = useRef<HTMLVideoElement>(null);
  const [count, setCount] = useState<number | null>(null);
  const levels = useLevel(r.stream, 12);

  useEffect(() => { if (preview.current) preview.current.srcObject = r.stream; }, [r.stream]);
  useEffect(() => { if (r.recording && r.elapsed >= VIDEO_SEC) r.stop(); }, [r.elapsed, r.recording]);
  useEffect(() => { if (r.result) r.close(); }, [r.result]);

  const go = async () => {
    const s = r.stream ?? (await r.open());
    if (!s) return;
    for (let n = 3; n > 0; n--) { setCount(n); await new Promise((ok) => setTimeout(ok, 800)); }
    setCount(null);
    r.start(s);
  };

  return (
    <div className="recorder">
      {!r.result ? (
        <>
          <div className="preview-wrap">
            <video ref={preview} className="video-preview" autoPlay muted playsInline />
            {r.stream && <div className="face-guide" />}
            {count !== null && <div className="countdown">{count}</div>}
          </div>
          {r.stream && <Meter levels={levels} />}
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            {!r.stream && <button className="btn" onClick={() => void r.open()}>Turn on camera</button>}
            {r.stream && !r.recording && count === null && <button className="btn primary" onClick={go}>Record {VIDEO_SEC} s</button>}
            {r.recording && <span className="mono">recording {clock(r.elapsed)} / {clock(VIDEO_SEC)}</span>}
          </div>
          <p className="muted small" style={{ margin: 0, textAlign: "center" }}>Face in the dashed oval, talk about anything. The training clips are 15 s webcam monologues.</p>
        </>
      ) : (
        <div style={{ display: "grid", gap: 12, justifyItems: "center" }}>
          <video controls src={r.result.url} className="video-preview" style={{ transform: "none" }} />
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn ghost" onClick={r.reset}>Record again</button>
            <button className="btn primary" onClick={() => onReady(r.result!.blob, `video${extFor(r.result!.mime)}`, r.result!.url, true)}>
              Run the pipeline →
            </button>
          </div>
        </div>
      )}
      {r.error && <div className="error-box">{r.error}</div>}
    </div>
  );
}

function UploadTab({ onReady }: { onReady: (blob: Blob, filename: string, url: string, isVideo: boolean) => void }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const take = (file?: File) => {
    if (!file) return;
    onReady(file, file.name, URL.createObjectURL(file), file.type.startsWith("video/"));
  };
  return (
    <div
      className={`dropzone ${over ? "over" : ""}`}
      onClick={() => input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}
    >
      <div style={{ fontSize: 16, color: "var(--ink)", marginBottom: 6 }}>Drop an audio or video file, or click to choose</div>
      <div className="small">mp3, wav, m4a, ogg, webm, mp4, mov · up to 25 MB and 3 minutes · longer inputs are cut into 15 s windows</div>
      <input ref={input} type="file" accept="audio/*,video/*" hidden onChange={(e) => take(e.target.files?.[0])} />
    </div>
  );
}

function SamplesTab({ onPick }: { onPick: (s: Sample) => void }) {
  const [samples, setSamples] = useState<Sample[] | null>(null);
  useEffect(() => { getSamples().then(setSamples).catch(() => setSamples([])); }, []);
  if (samples === null) return <div className="muted">loading…</div>;
  if (samples.length === 0)
    return <div className="note-box">No samples yet. Make one with <span className="mono">python src/predict.py voice.mp3 --record my-sample</span>.</div>;
  return (
    <div className="sample-list">
      <p className="ink2" style={{ margin: 0 }}>
        Recorded runs, replayed with their real timing. They work without a microphone and without waiting for the models.
      </p>
      {samples.map((s) => (
        <button key={s.name} className="sample" onClick={() => onPick(s)}>
          <strong>{s.title}</strong>
          <span className="small ink2">{s.description}</span>
          <span className="tiny muted mono">{s.video ? "video + voice" : "voice only"} · ran in {s.seconds?.toFixed(1)} s</span>
        </button>
      ))}
    </div>
  );
}

export function Capture({ onStart }: { onStart: (input: RunInput) => void }) {
  const [tab, setTab] = useState<Tab>("voice");
  const [consent, setConsent] = useState(false);
  const live = tab !== "samples";
  const tabs: { id: Tab; label: string; hint: string }[] = [
    { id: "voice", label: "Record voice", hint: "microphone · 5–30 s" },
    { id: "video", label: "Record video", hint: "webcam · 15 s · all 3 lanes" },
    { id: "upload", label: "Upload a file", hint: "audio or video" },
    { id: "samples", label: "Demo samples", hint: "no mic needed" },
  ];
  const ready = (blob: Blob, filename: string, mediaUrl: string, isVideo: boolean) =>
    onStart({ kind: "live", blob, filename, mediaUrl, isVideo, title: filename.startsWith("voice") ? "Your voice" : filename.startsWith("video") ? "Your video" : filename });

  return (
    <div className="capture">
      <div className="capture-tabs" role="tablist">
        {tabs.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>
            {t.label}<span>{t.hint}</span>
          </button>
        ))}
      </div>
      <div className="capture-body">
        {live && !consent ? (
          <div className="note-box" style={{ alignSelf: "stretch" }}>
            Tick the box below to turn on {tab === "upload" ? "uploads" : tab === "video" ? "the camera" : "the microphone"}.
          </div>
        ) : tab === "voice" ? <VoiceTab onReady={ready} />
          : tab === "video" ? <VideoTab onReady={ready} />
          : tab === "upload" ? <UploadTab onReady={ready} />
          : <SamplesTab onPick={(sample) => onStart({ kind: "sample", sample })} />}
        {live && (
          <label className="consent">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            <span>
              I understand: the recording is sent to this computer's local server, analysed, and deleted as soon as the run ends.
              Nothing is stored. The result is an <em>apparent</em> first impression from a research model, not a personality test.
            </span>
          </label>
        )}
      </div>
    </div>
  );
}
