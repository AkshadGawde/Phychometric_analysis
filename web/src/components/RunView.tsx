import { useEffect, useRef } from "react";
import { usePlayhead } from "../lib/playhead";
import { useRun } from "../lib/run";
import { FaceLane } from "./FaceLane";
import { FusionLane } from "./FusionLane";
import { PipelineMap } from "./PipelineMap";
import { Result } from "./Result";
import { VoiceLane } from "./VoiceLane";
import { WordsLane } from "./WordsLane";
import { clock } from "./charts";

export interface RunSpec {
  eventsUrl: string;
  mediaUrl: string | null;
  isVideo: boolean;
  title: string;
  replay: boolean;
}

function Player({ spec, fallbackDuration }: { spec: RunSpec; fallbackDuration: number }) {
  const { register, time, playing, duration, toggle, seek } = usePlayhead();
  const media = useRef<HTMLMediaElement | null>(null);
  const total = duration || fallbackDuration;   // browser recordings often report no duration
  useEffect(() => () => { media.current?.pause(); }, []);
  const ref = (el: HTMLMediaElement | null) => { media.current = el; register(el); };
  return (
    <div className="player">
      <button className="play" onClick={toggle} disabled={!spec.mediaUrl} aria-label={playing ? "Pause" : "Play"}>
        {playing ? "❚❚" : "▶"}
      </button>
      {spec.mediaUrl && (spec.isVideo
        ? <video ref={ref} src={spec.mediaUrl} playsInline preload="auto" />
        : <audio ref={ref} src={spec.mediaUrl} preload="auto" />)}
      <span className="mono small">{clock(time)}</span>
      <div className="scrub" onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); total && seek(((e.clientX - r.left) / r.width) * total); }}>
        <div className="scrub-track" />
        <div className="scrub-fill" style={{ width: total ? `${(time / total) * 100}%` : 0 }} />
        <div className="scrub-knob" style={{ left: total ? `${Math.min(100, (time / total) * 100)}%` : 0 }} />
      </div>
      <span className="mono small muted">{clock(total)}</span>
      <span className="small muted" style={{ maxWidth: 220 }}>play to watch every lane follow the sound</span>
    </div>
  );
}

export function RunView({ spec, onNew }: { spec: RunSpec; onNew: () => void }) {
  const { state, data } = useRun(spec.eventsUrl);
  const ingest = data("ingest");
  const fusion = data("fusion");

  return (
    <div>
      <div className="run-head">
        <div>
          <span className="upper muted">{spec.replay ? "Replay of a recorded run" : "Live run"}</span>
          <h2>{spec.title}</h2>
        </div>
        <div className="chips" style={{ marginTop: 0 }}>
          {ingest && <span className="chip"><b>{ingest.raw_sec.toFixed(1)} s</b> · {ingest.container.split(",")[0]} · {ingest.codecs.join(", ")}</span>}
          {ingest && <span className="chip">{ingest.has_video ? "video + sound: all 3 lanes" : "sound only: face lane skipped"}</span>}
          {state.phase === "done" && <span className="chip">ran in <b>{state.total?.toFixed(1)} s</b></span>}
        </div>
        <button className="btn" style={{ marginLeft: "auto" }} onClick={onNew}>New input</button>
      </div>

      {state.phase === "connecting" && <div className="note-box" style={{ marginBottom: 16 }}>Uploading and connecting…</div>}
      {state.phase === "queued" && state.queue && (
        <div className="note-box" style={{ marginBottom: 16 }}>
          {state.queue.ready
            ? state.queue.position > 0 ? `Waiting: ${state.queue.position} run(s) ahead of you (one runs at a time).` : "Starting…"
            : `The server is still getting ready (${state.queue.status}). Your run starts right after.`}
        </div>
      )}
      {state.error && <div className="error-box" style={{ marginBottom: 16 }}>Stopped at “{state.error.stage}”: {state.error.message}</div>}

      <Player spec={spec} fallbackDuration={ingest?.raw_sec ?? 0} />
      <PipelineMap state={state} />
      <VoiceLane state={state} clean={data("audio_clean")} features={data("audio_features")} />
      <WordsLane state={state} data={data("transcribe")} />
      <FaceLane state={state} data={data("visual")} duration={ingest?.raw_sec ?? 0} />
      <FusionLane state={state} uni={data("unimodal")} fusion={fusion} />
      {fusion && <Result fusion={fusion} explain={data("explain")} />}
    </div>
  );
}
