import type { CSSProperties } from "react";
import { usePlayhead } from "../lib/playhead";
import type { RunState } from "../lib/run";
import { TRAIT_LABEL, type Token, type TranscribeData } from "../types";
import { Card, LANE_VAR } from "./Stage";
import { ProbBars } from "./charts";

function cueClass(t: Token) {
  if (!t.cues.length) return "";
  const sides = new Set(t.cues.map((c) => c.side));
  return sides.size > 1 ? "cue-mixed" : sides.has("high") ? "cue-up" : "cue-down";
}

function cueTitle(t: Token) {
  if (t.removed) return "removed in cleaning (filler / punctuation)";
  if (!t.cues.length) return t.p !== undefined ? `recognised with ${(t.p * 100).toFixed(0)}% confidence` : "";
  return t.cues.map((c) => `${TRAIT_LABEL[c.trait]}: more common in clips rated ${c.side}`).join("\n");
}

function Transcript({ d }: { d: TranscribeData }) {
  const { time, seek } = usePlayhead();
  return (
    <>
      <div className="transcript">
        {d.tokens.map((t, i) => {
          const now = t.start !== null && t.end !== null && time >= t.start && time < t.end + 0.05;
          const lowConf = t.p !== undefined && t.p < 0.5;
          return (
            <span key={i} className={`tok ${t.removed ? "removed" : ""} ${cueClass(t)} ${now ? "now" : ""} ${lowConf ? "low-conf" : ""}`}
              style={{ animationDelay: `${Math.min(i, 120) * 18}ms`, cursor: t.start !== null ? "pointer" : undefined }}
              title={cueTitle(t)} onClick={() => t.start !== null && seek(t.start)}>
              {t.w}
            </span>
          );
        })}
      </div>
      <div className="clean-text" aria-label="text after cleaning">{d.clean || "—"}</div>
      <div className="chips">
        <span className="chip"><b>{d.n_words}</b> words</span>
        <span className="chip"><b>{d.words_per_min.toFixed(0)}</b> words / min</span>
        <span className="chip">source <b>{d.source}</b></span>
        <span className="chip" style={{ borderColor: "var(--up)" }}>underlined green: cue for "high"</span>
        <span className="chip" style={{ borderColor: "var(--down)" }}>red: cue for "low"</span>
      </div>
      <div className="caption">
        Top: what Whisper heard, word by word (click a word to hear it). Bottom: after cleaning (lower case, fillers and
        punctuation removed), the form the word-cue analysis uses. The two text models read the transcript with punctuation.
      </div>
    </>
  );
}

export function WordsLane({ state, data }: { state: RunState; data?: TranscribeData }) {
  const skipped = data && !data.available;
  return (
    <div className="lane" style={{ "--lane": LANE_VAR.words } as CSSProperties}>
      <div className="lane-head">
        <h3><i />Words</h3>
        <span className="small muted">the weakest lane in the study (test AUC 0.55, close to chance)</span>
      </div>
      <div className="lane-grid">
        <Card lane="words" step="W1" title="Speech recognition → cleaning" stage={state.stages.transcribe} ready={!!data} className="span-8"
          busyText="Whisper is listening (runs at the same time as the voice lane)…">
          {data && (skipped ? <div className="note-box">{data.note}</div> : <Transcript d={data} />)}
        </Card>
        <Card lane="words" step="W2" title="Text emotion · DistilRoBERTa" stage={state.stages.transcribe} ready={!!data} className="span-4">
          {data && (data.emotion ? (
            <>
              <ProbBars labels={data.emotion_labels} values={data.emotion} />
              <div className="caption">+ a {data.embedding_dims}-number meaning embedding (MiniLM) of the whole transcript, also fed to the model.</div>
            </>
          ) : <div className="note-box">no words, so no text features: this lane is left out of the fusion</div>)}
        </Card>
      </div>
    </div>
  );
}
