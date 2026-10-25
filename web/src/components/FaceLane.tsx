import type { CSSProperties } from "react";
import { usePlayhead } from "../lib/playhead";
import type { RunState } from "../lib/run";
import type { VisualData } from "../types";
import { Card, LANE_VAR } from "./Stage";
import { ProbBars } from "./charts";

const box = (b: number[]) => ({ left: `${b[0] * 100}%`, top: `${b[1] * 100}%`, width: `${b[2] * 100}%`, height: `${b[3] * 100}%` });

function Frames({ d, duration }: { d: VisualData; duration: number }) {
  const { time, seek } = usePlayhead();
  const frames = d.frames ?? [];
  const n = frames.length;
  const at = (k: number) => (n > 1 ? (k / (n - 1)) * duration : 0);
  const current = duration > 0 && time > 0 ? Math.round((time / duration) * (n - 1)) : -1;
  const labels = d.emotion_labels ?? [];
  return (
    <>
      <div className="frames">
        {frames.map((f, k) => {
          const top = f.emotion.indexOf(Math.max(...f.emotion));
          return (
            <div key={k} className="frame" style={{ animationDelay: `${k * 70}ms` }} onClick={() => seek(at(k))}>
              <div className="frame-full" style={{ outline: k === current ? "1.5px solid #fff" : undefined, cursor: "pointer" }}>
                <img src={f.frame} alt={`frame ${k + 1}`} />
                <div className="crop" style={box(f.crop_box)} />
                {f.face && <div className="box" style={{ ...box(f.face), animationDelay: `${300 + k * 70}ms` }} />}
              </div>
              <img className="crop-img" src={f.crop} alt={`face crop ${k + 1}`} style={{ animationDelay: `${450 + k * 70}ms` }} />
              <div className="tag">{f.detected ? (f.replaced ? "median box" : "face") : "centre crop"} · <b>{labels[top]}</b></div>
            </div>
          );
        })}
      </div>
      <div className="chips">
        <span className="chip">faces found <b>{d.faces_found}/{d.n_frames}</b></span>
        {!!d.replaced && <span className="chip">outliers replaced <b>{d.replaced}</b></span>}
        <span className="chip">black bars <b>{d.bars_removed ? "removed" : "none"}</b></span>
        <span className="chip">ResNet-50 <b>{d.cnn_dims}</b> numbers / frame</span>
      </div>
      <div className="caption">8 evenly spaced frames → black bars cut → largest face (Haar cascade, amber box) + 20% margin (dashed) → 224×224 crop. Features are the mean over the 8 crops.</div>
    </>
  );
}

export function FaceLane({ state, data, duration }: { state: RunState; data?: VisualData; duration: number }) {
  return (
    <div className="lane" style={{ "--lane": LANE_VAR.face } as CSSProperties}>
      <div className="lane-head">
        <h3><i />Face</h3>
        <span className="small muted">test AUC 0.66 in the study · the lane that overfits the most</span>
      </div>
      <div className="lane-grid">
        <Card lane="face" step="F1" title="Frames → face crops" stage={state.stages.visual} ready={!!data?.available} className="span-8"
          busyText="face detection, ResNet-50, ViT…" skippedText={data?.reason ?? "skipped"}>
          {data?.available && <Frames d={data} duration={duration} />}
        </Card>
        <Card lane="face" step="F2" title="Face expression · ViT" stage={state.stages.visual} ready={!!data?.available} className="span-4"
          skippedText="no face lane for audio-only input: record a video to light it up">
          {data?.available && data.emotion && (
            <>
              <ProbBars labels={data.emotion_labels ?? []} values={data.emotion} />
              <div className="caption">Mean over the 8 crops; these 7 numbers + the 2,048 ResNet numbers go into the face model.</div>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
