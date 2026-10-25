import type { CSSProperties } from "react";
import type { RunState } from "../lib/run";
import { LANE_LABEL, MODALITIES, TRAITS, TRAIT_LABEL, type FusionData, type Modality, type UnimodalData } from "../types";
import { Card, LANE_VAR } from "./Stage";
import { ordinal, pct } from "./charts";

const LANE_OF: Record<Modality, string> = { visual: "face", audio: "voice", text: "words" };

/** Per trait: each lane's P(high) -> the fusion node (line thickness = weight) -> fused value. */
function FuseRows({ uni, fusion }: { uni: UnimodalData; fusion?: FusionData }) {
  return (
    <div className="fusion-grid">
      {TRAITS.map((trait) => {
        const fused = fusion?.traits[trait];
        return (
          <div key={trait} className="fuse-row">
            <div>
              <div style={{ fontWeight: 600, fontSize: 14 }}>{TRAIT_LABEL[trait]}</div>
              <div className="tiny muted mono">{fused ? `from ${fused.subset.replace(/\+/g, " + ")}` : "…"}</div>
            </div>
            <div className="fuse-inputs">
              {MODALITIES.map((m) => {
                const p = uni.prob[m]?.[trait];
                const w = fused?.weights[m];
                const off = p === undefined || (fused !== undefined && w === undefined);
                return (
                  <div key={m} className={`fuse-in ${off ? "off" : ""}`} style={{ "--lane": LANE_VAR[LANE_OF[m]] } as CSSProperties}>
                    <span className="ink2">{LANE_LABEL[m]}</span>
                    <span className="track"><i style={{ width: p === undefined ? 0 : pct(p, 1), background: LANE_VAR[LANE_OF[m]] }} /></span>
                    <span className="mono small">{p === undefined ? "—" : p.toFixed(2)}</span>
                    <span className="w">{p === undefined ? "no input" : w === undefined ? (fused ? "unused" : "") : `w ${w.toFixed(2)}`}</span>
                  </div>
                );
              })}
            </div>
            <svg className="fuse-node" viewBox="0 0 70 54" aria-hidden>
              {MODALITIES.map((m, i) => {
                const w = fused?.weights[m] ?? 0;
                const has = uni.prob[m]?.[trait] !== undefined;
                const y = 9 + i * 18;
                return (
                  <path key={m} d={`M0,${y} C30,${y} 30,27 56,27`} fill="none" stroke={LANE_VAR[LANE_OF[m]]}
                    strokeWidth={fused ? 0.5 + w * 6 : has ? 1 : 0.6} strokeOpacity={has ? (fused && !w ? 0.15 : 0.9) : 0.12}
                    strokeDasharray={has ? undefined : "2 3"} style={{ transition: "stroke-width 0.6s ease" }} />
                );
              })}
              <circle cx="60" cy="27" r="7" fill={fused ? "var(--fusion)" : "var(--surface-3)"} style={{ transition: "fill 0.4s" }} />
            </svg>
            <div className="fused-value">
              {fused ? (
                <>
                  <span className="big" style={{ color: "var(--fusion)" }}>{fused.prob.toFixed(2)}</span>
                  <span className="tiny muted">P(rated high) · {ordinal(fused.percentile)} percentile</span>
                </>
              ) : <span className="muted small">waiting for fusion</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function FusionLane({ state, uni, fusion }: { state: RunState; uni?: UnimodalData; fusion?: FusionData }) {
  const skipped = uni?.skipped ?? [];
  return (
    <div className="lane" style={{ "--lane": LANE_VAR.fusion } as CSSProperties}>
      <div className="lane-head">
        <h3><i />Fusion</h3>
        <span className="small muted">one trained model per lane and trait, then a weighted average (weights = validation AUC)</span>
      </div>
      <div className="lane-grid">
        <Card lane="fusion" step="M1" title={fusion ? "Late fusion (weighted)" : "One model per lane"} stage={fusion ? state.stages.fusion : state.stages.unimodal}
          ready={!!uni} className="span-12" busyText="scaler → PCA → classifier, per lane and trait…">
          {uni && (
            <>
              {skipped.length > 0 && (
                <div className="note-box" style={{ marginBottom: 10 }}>
                  {skipped.map((m) => LANE_LABEL[m]).join(" and ")} missing → {skipped.length > 1 ? "those lanes are" : "that lane is"} disconnected.{" "}
                  {fusion?.choice_rule ? `Rule: ${fusion.choice_rule}.` : ""}
                </div>
              )}
              <FuseRows uni={uni} fusion={fusion} />
              <div className="caption">
                Bars: each lane's own probability that strangers would rate this person above the median (0.5 = undecided).
                {fusion && ` Percentile: position among the ${fusion.held_out_clips} held-out clips (validation + test) scored by the same fusion.`}
              </div>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}
