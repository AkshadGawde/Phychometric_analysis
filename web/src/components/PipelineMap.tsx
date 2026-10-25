import type { CSSProperties } from "react";
import type { RunState, StageState } from "../lib/run";
import type { StageName } from "../types";
import { LANE_VAR, StageTimer } from "./Stage";

// The overview: one node per REAL server stage (no extra decorative steps).

function Node({ stage, label, lane }: { stage: StageState; label: string; lane: string }) {
  return (
    <div className={`node ${stage.status}`} style={{ "--lane": LANE_VAR[lane] } as CSSProperties}>
      <span>{label}</span>
      <StageTimer stage={stage} />
    </div>
  );
}

export function PipelineMap({ state }: { state: RunState }) {
  const s = (name: StageName) => state.stages[name];
  const lanes: { lane: string; label: string; nodes: [StageName, string][] }[] = [
    { lane: "voice", label: "Voice", nodes: [["ingest", "decode 16 kHz"], ["audio_clean", "denoise + trim"], ["audio_features", "MFCC · pitch · wav2vec2"]] },
    { lane: "words", label: "Words", nodes: [["transcribe", "Whisper → MiniLM + emotion"]] },
    { lane: "face", label: "Face", nodes: [["visual", "8 frames → face → ResNet + ViT"]] },
  ];
  const fusionDone = (n: StageName) => s(n).status;
  return (
    <div className="map">
      <div className="map-lanes">
        {lanes.map((l) => (
          <div key={l.lane} className="map-lane" style={{ "--lane": LANE_VAR[l.lane] } as CSSProperties}>
            <span className="upper">{l.label}</span>
            {l.nodes.map(([name, label]) => (
              <div key={name} style={{ gridColumn: l.nodes.length === 1 ? "span 3" : undefined }}>
                <Node stage={s(name)} label={label} lane={l.lane} />
              </div>
            ))}
          </div>
        ))}
      </div>
      <svg className="map-join" viewBox="0 0 70 120" aria-hidden>
        {[20, 60, 100].map((y, i) => (
          <path key={y} d={`M0,${y} C35,${y} 35,60 70,60`} fill="none"
            stroke={["var(--voice)", "var(--words)", "var(--face)"][i]}
            strokeOpacity={fusionDone("unimodal") === "done" ? 0.9 : 0.25} strokeWidth="1.5" />
        ))}
      </svg>
      <div className="map-out">
        <Node stage={s("unimodal")} label="one model per lane" lane="fusion" />
        <Node stage={s("fusion")} label="late fusion (weighted)" lane="fusion" />
        <Node stage={s("explain")} label="explain" lane="fusion" />
      </div>
    </div>
  );
}
