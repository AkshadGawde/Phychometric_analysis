import { useMemo, useState, type CSSProperties } from "react";
import { percentile, refuse, subsetKey, type Refused } from "../lib/fusion";
import { LANE_LABEL, MODALITIES, TRAITS, TRAIT_LABEL, type ExplainData, type FusionData, type Modality, type Trait } from "../types";
import { LANE_VAR } from "./Stage";
import { ordinal } from "./charts";

const LANE_OF: Record<Modality, string> = { visual: "face", audio: "voice", text: "words" };

const describe = (p: number) =>
  p >= 80 ? "well above most clips" : p >= 60 ? "above most clips" : p > 40 ? "around the middle" : p > 20 ? "below most clips" : "well below most clips";

function Scale({ value, band }: { value: number; band: [number, number] | null }) {
  return (
    <div className="scale">
      <div className="axis" />
      <div className="mid" />
      {band && <div className="band" style={{ left: `${band[0]}%`, width: `${Math.max(1, band[1] - band[0])}%` }} />}
      <div className="pin" style={{ left: `${value}%` }} />
      <div className="ticks"><span>0</span><span>50th</span><span>100th percentile</span></div>
    </div>
  );
}

function Reliability({ r }: { r: Refused }) {
  return (
    <div className="reliability">
      <div>Right <b className="mono" style={{ color: "var(--ink)" }}>{(r.test.acc * 100).toFixed(0)}%</b> of the time on unseen test clips (chance 50%)</div>
      <div className="rel-bar" title="accuracy on the 221 test clips"><i style={{ width: `${r.test.acc * 100}%` }} /><b /></div>
      <div className="tiny muted mono">AUC {r.test.auc.toFixed(2)} · F1 {r.test.f1.toFixed(2)} · {r.subset.replace(/\+/g, " + ")}</div>
    </div>
  );
}

export function Result({ fusion, explain }: { fusion: FusionData; explain?: ExplainData }) {
  const available = fusion.available;
  // null = the default rule (per trait, validation choice); otherwise the same subset for all traits
  const [chosen, setChosen] = useState<Set<Modality> | null>(null);

  const rows = useMemo(() => TRAITS.map((trait) => {
    const key = chosen ? subsetKey(chosen) : fusion.traits[trait].subset;
    const r = refuse(fusion, trait, key);
    const spread = !chosen ? fusion.traits[trait].spread : null;
    const option = fusion.options[key]?.[trait];
    const band = spread && option ? [percentile(spread[0], option.quantiles), percentile(spread[1], option.quantiles)] as [number, number] : null;
    return { trait, r, band };
  }), [fusion, chosen]);

  const toggle = (m: Modality) => {
    const next = new Set(chosen ?? available);
    if (next.has(m)) next.delete(m); else next.add(m);
    if (next.size === 0) return;
    setChosen(next);
  };

  return (
    <section className="result">
      <span className="upper muted">Result</span>
      <h2>How strangers would probably rate this first impression</h2>
      <p className="ink2" style={{ margin: "4px 0 0", maxWidth: 760 }}>
        Each trait: where this input lands among the {fusion.held_out_clips} held-out clips of the study, as rated by crowd workers who watched 15 seconds.
        It is <em>apparent</em> personality, not a measurement of the person.
      </p>

      <div className="toggles">
        <span className="small muted" style={{ marginRight: 4 }}>Live ablation:</span>
        <button className={`toggle ${chosen === null ? "on" : ""}`} style={{ "--lane": "var(--fusion)" } as CSSProperties} onClick={() => setChosen(null)}>
          <i />validation choice
        </button>
        {MODALITIES.map((m) => {
          const has = available.includes(m);
          const on = has && (chosen ? chosen.has(m) : false);
          return (
            <button key={m} className={`toggle ${on ? "on" : ""}`} disabled={!has} style={{ "--lane": LANE_VAR[LANE_OF[m]] } as CSSProperties}
              onClick={() => toggle(m)} title={has ? "use / drop this lane for all traits" : "no input for this lane"}>
              <i />{LANE_LABEL[m]}
            </button>
          );
        })}
        <span className="tiny muted">{chosen ? `fusing ${[...chosen].map((m) => LANE_LABEL[m]).join(" + ")} for every trait, weights renormalized` : fusion.choice_rule}</span>
      </div>

      {rows.map(({ trait, r, band }) => r && (
        <div key={trait} className="trait">
          <div>
            <h3>{TRAIT_LABEL[trait]}</h3>
            <div className="mono" style={{ fontSize: 26, fontWeight: 600, letterSpacing: "-0.02em", marginTop: 4 }}>{ordinal(r.percentile)}</div>
            <div className="small ink2">{describe(r.percentile)}</div>
            {trait === "neuroticism" && <div className="tiny muted" style={{ marginTop: 4 }}>labelled "neuroticism" in the dataset, but it is emotional stability: high = calm</div>}
          </div>
          <div>
            <Scale value={r.percentile} band={band} />
            <Why trait={trait} explain={explain} r={r} fusion={fusion} />
          </div>
          <Reliability r={r} />
        </div>
      ))}

      <div className="fineprint">
        The band (when shown) is the spread between the ~15 s windows of a long input. "Right x% of the time" is the accuracy
        of exactly this combination of lanes on the 221 test clips nobody trained on. Research prototype: never use this to judge,
        hire, grade or diagnose anyone. The labels are first impressions of strangers, with their biases (age, gender, ethnicity,
        looks), which this study did not test.
      </div>
    </section>
  );
}

function Why({ trait, explain, r, fusion }: { trait: Trait; explain?: ExplainData; r: Refused; fusion: FusionData }) {
  const e = explain?.traits[trait];
  const parts = (Object.entries(r.weights) as [Modality, number][]).map(([m, w]) => ({ m, v: w * ((fusion.unimodal_prob[m]?.[trait] ?? 0.5) - 0.5) }));
  const total = parts.reduce((s, p) => s + Math.abs(p.v), 0) || 1;
  return (
    <div className="why">
      <div className="contrib" title="how far each lane pulls away from 0.5 (undecided)">
        {parts.map((p) => <span key={p.m} style={{ width: `${(Math.abs(p.v) / total) * 100}%`, background: LANE_VAR[LANE_OF[p.m]], opacity: p.v >= 0 ? 1 : 0.45 }} />)}
      </div>
      <div className="tiny muted">
        pull away from 0.5: {parts.map((p) => `${LANE_LABEL[p.m]} ${p.v >= 0 ? "▲" : "▼"} ${Math.abs(p.v).toFixed(2)}`).join(" · ")}
      </div>
      {e && e.voice_cues.slice(0, 2).map((c) => (
        <div key={c.feature}>
          <span className={c.direction === "up" ? "arrow-up" : "arrow-down"}>{c.direction === "up" ? "▲" : "▼"}</span>{" "}
          {c.label}{c.percentile !== undefined ? ` (${ordinal(c.percentile)} percentile)` : ` (${c.z > 0 ? "+" : ""}${c.z.toFixed(1)} SD)`}
          <span className="muted"> {c.direction === "up" ? "reads as higher" : "reads as lower"}</span>
        </div>
      ))}
      {e?.words && (e.words.high.length > 0 || e.words.low.length > 0) && (
        <div>
          words: {e.words.high.slice(0, 4).map((w) => <span key={w} className="arrow-up">“{w}” </span>)}
          {e.words.low.slice(0, 4).map((w) => <span key={w} className="arrow-down">“{w}” </span>)}
        </div>
      )}
    </div>
  );
}
