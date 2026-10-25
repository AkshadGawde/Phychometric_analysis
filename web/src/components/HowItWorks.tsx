import { useEffect, useState, type CSSProperties } from "react";
import { getModelCard } from "../lib/api";
import { TRAITS, TRAIT_LABEL } from "../types";

// The study itself, from results/*.csv (through /api/model-card). Every number here is the same
// as in results/REPORT.md.

type Metric = "auc" | "acc" | "f1";
const METRIC_LABEL: Record<Metric, string> = { auc: "AUC", acc: "accuracy", f1: "F1" };
const COLOR: Record<string, string> = {
  visual: "var(--face)", audio: "var(--voice)", text: "var(--words)", text_tfidf: "var(--words)",
  early: "var(--fusion)", late: "var(--fusion)", late_weighted: "var(--fusion)", intermediate: "var(--fusion)",
};

function CiChart({ rows, metric, chosen }: { rows: any[]; metric: Metric; chosen: string }) {
  const lo = metric === "f1" ? 0.4 : 0.4, hi = metric === "f1" ? 0.75 : 0.8;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  const grid = [0.4, 0.5, 0.6, 0.7, 0.8].filter((g) => g <= hi);
  return (
    <div className="ci-chart">
      {rows.filter((r) => r[metric]).map((r) => {
        const [est, a, b] = r[metric];
        const color = COLOR[r.model] ?? "var(--muted)";
        return (
          <div key={r.model} className={`ci-row ${r.model === chosen ? "chosen" : ""}`}>
            <span>{r.label}{r.model === chosen ? " · chosen on validation" : ""}</span>
            <div className="ci-track">
              {grid.map((g) => <span key={g} className="grid" style={{ left: x(g) }} />)}
              {a !== undefined && <span className="whisker" style={{ left: x(a), width: `calc(${x(b)} - ${x(a)})`, background: color, opacity: 0.55 }} />}
              <span className="pt" style={{ left: x(est), background: color }} />
            </div>
            <span className="mono small ink2">{est.toFixed(3)}{a !== undefined ? ` [${a.toFixed(2)}, ${b.toFixed(2)}]` : ""}</span>
          </div>
        );
      })}
      <div className="ci-row tiny muted"><span /><div className="ci-track" style={{ height: 12 }}>
        {grid.map((g) => <span key={g} className="mono" style={{ position: "absolute", left: x(g), transform: "translateX(-50%)" }}>{g.toFixed(1)}</span>)}
      </div><span /></div>
    </div>
  );
}

function Heat({ rows, cols, matrix, rowLabel }: { rows: string[]; cols: string[]; matrix: number[][]; rowLabel: (r: string) => string }) {
  const cell = (v: number) => {
    const a = Math.min(1, Math.abs(v));
    return v >= 0 ? `rgba(52,211,153,${a * 0.75})` : `rgba(251,113,133,${a * 0.75})`;
  };
  return (
    <div className="scroll-x">
      <table className="mini heat">
        <thead><tr><th />{cols.map((c) => <th key={c} style={{ textAlign: "center" }}>{TRAIT_LABEL[c as keyof typeof TRAIT_LABEL] ?? c}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={r}><td>{rowLabel(r)}</td>{matrix[i].map((v, j) => <td key={j} className="cell" style={{ background: cell(v) }}>{v.toFixed(2)}</td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export function HowItWorks() {
  const [card, setCard] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<Metric>("auc");
  useEffect(() => { getModelCard().then(setCard).catch((e) => setError(String(e))); }, []);
  if (error) return <div className="error-box">{error}. Is the API running?</div>;
  if (!card) return <div className="muted">loading the study…</div>;

  const d = card.dataset;
  // "Is fusion better than the best single lane?" from results/diagnostics/fusion_vs_single.csv
  const gain = (card.fusion_vs_single as any[]).find((g) => g.metric === "auc");
  const fusionVsVoice = gain
    ? `Fusion vs ${gain.single === "audio" ? "voice" : gain.single} alone: AUC ${gain.gain >= 0 ? "+" : ""}${gain.gain.toFixed(3)} ` +
      `(95% CI ${gain.gain_ci_low.toFixed(3)} to ${gain.gain_ci_high.toFixed(3)}), ` +
      (gain.gain_significant ? "a real difference." : "inside the noise: the voice alone is about as good.")
    : "";
  const majorityF1: number | undefined = (card.table1 as any[]).find((r) => r.model === "baseline: majority class")?.f1?.[0];
  const subsetRows = Object.entries(card.subsets as Record<string, Record<string, { auc: number; acc: number }>>)
    .map(([k, v]) => ({ k, auc: TRAITS.reduce((s, t) => s + v[t].auc, 0) / TRAITS.length, acc: TRAITS.reduce((s, t) => s + v[t].acc, 0) / TRAITS.length }))
    .sort((a, b) => b.auc - a.auc);
  const curve = card.learning_curve as { train_clips: number; test_auc: number; test_f1: number }[];
  const cx = (n: number) => 20 + ((n - 200) / (1100 - 200)) * 360, cy = (v: number) => 110 - ((v - 0.6) / 0.15) * 100;

  return (
    <div className="how">
      <span className="upper muted">How it works</span>
      <h1>The study behind the visualizer</h1>
      <p>
        Can a machine copy the first impression strangers form from a 15-second video? The data is ChaLearn First Impressions V2:
        YouTube monologues rated by crowd workers on the Big Five. We used {d.clips.toLocaleString()} clips from {d.youtube_videos.toLocaleString()} videos,
        split by video so the same person never appears in two parts.
      </p>
      <div className="stat-row">
        {(["train", "val", "test"] as const).map((s) => (
          <div key={s} className="stat"><div className="upper muted">{s === "val" ? "validation" : s}</div><div className="v">{d.splits[s].clips}</div><div className="small muted">clips · {d.splits[s].videos} videos</div></div>
        ))}
        <div className="stat"><div className="upper muted">label-shuffle test</div><div className="v">{card.shuffle_test.test_auc.toFixed(2)}</div><div className="small muted">AUC when trained on shuffled labels: chance, so no leakage</div></div>
      </div>

      <section>
        <h2>The pipeline</h2>
        <p>Each lane turns its input into numbers with frozen, pretrained networks, then one small model per lane and trait is trained on those numbers. Late fusion averages the lanes, weighted by their validation AUC.</p>
        <div className="scroll-x"><table className="mini">
          <thead><tr><th>lane</th><th>preprocessing</th><th>features</th><th>model</th></tr></thead>
          <tbody>
            <tr><td style={{ color: "var(--voice)" }}>Voice</td><td>16 kHz mono, spectral gating, silence trim, peak normalisation</td><td>87 hand-crafted (40 MFCC means + spreads, pitch, loudness, voicing) + 8 emotions ({card.models.speech_emotion})</td><td>logistic regression / MLP</td></tr>
            <tr><td style={{ color: "var(--words)" }}>Words</td><td>transcript (Whisper {card.whisper} in the app), notes removed</td><td>384-d MiniLM embedding + 7 emotions ({card.models.text_emotion})</td><td>logistic regression / MLP</td></tr>
            <tr><td style={{ color: "var(--face)" }}>Face</td><td>8 frames, black bars cut, Haar face crop + 20% margin, 224×224</td><td>2,048-d ResNet-50 + 7 expressions ({card.models.face_emotion}), mean over frames</td><td>scaler → PCA 95% → logistic regression / MLP</td></tr>
            <tr><td style={{ color: "var(--fusion)" }}>Fusion</td><td colSpan={2}>late fusion, weights = validation AUC per lane and trait (also tried: early, average, intermediate network)</td><td>chosen on validation: {card.chosen_fusion.replace("_", " ")}</td></tr>
          </tbody>
        </table></div>
      </section>

      <section>
        <h2>Results on the 221 test clips</h2>
        <p>Mean over the five traits, with 95% bootstrap intervals. {fusionVsVoice}</p>
        <div className="toggles">
          {(Object.keys(METRIC_LABEL) as Metric[]).map((m) => (
            <button key={m} className={`toggle ${metric === m ? "on" : ""}`} style={{ "--lane": "var(--ink-2)" } as CSSProperties} onClick={() => setMetric(m)}><i />{METRIC_LABEL[m]}</button>
          ))}
          {metric === "f1" && majorityF1 !== undefined && <span className="tiny muted">careful: answering "high" for everyone already gets F1 {majorityF1.toFixed(2)}</span>}
        </div>
        <CiChart rows={card.table1} metric={metric} chosen={card.chosen_fusion} />
      </section>

      <section>
        <h2>Ablation: every combination of lanes</h2>
        <p>The same weighted late fusion on each subset of lanes (mean test AUC over the traits). This is exactly what the toggles on the result screen do.</p>
        <div className="ci-chart">
          {subsetRows.map((r) => (
            <div key={r.k} className="ci-row">
              <span>{r.k.replace("visual", "Face").replace("audio", "Voice").replace("text", "Words").replace(/\+/g, " + ")}</span>
              <div className="ci-track"><span className="whisker" style={{ left: 0, width: `${((r.auc - 0.5) / 0.3) * 100}%`, background: "var(--fusion)", top: 6, height: 6 }} /></div>
              <span className="mono small ink2">AUC {r.auc.toFixed(3)} · acc {r.acc.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>More data would help, a little</h2>
        <p>Test AUC of the chosen model when trained on 25%, 50%, 75% and 100% of the training clips.</p>
        <svg viewBox="0 0 400 130" style={{ width: "min(100%, 520px)" }}>
          {[0.6, 0.65, 0.7, 0.75].map((v) => <g key={v}><line x1="20" x2="380" y1={cy(v)} y2={cy(v)} stroke="var(--line)" /><text x="0" y={cy(v) + 4} fill="var(--muted)" fontSize="9" fontFamily="var(--mono)">{v.toFixed(2)}</text></g>)}
          <polyline fill="none" stroke="var(--fusion)" strokeWidth="2" points={curve.map((p) => `${cx(p.train_clips)},${cy(p.test_auc)}`).join(" ")} />
          {curve.map((p) => <g key={p.train_clips}><circle cx={cx(p.train_clips)} cy={cy(p.test_auc)} r="3.5" fill="var(--fusion)" /><text x={cx(p.train_clips)} y="126" textAnchor="middle" fill="var(--muted)" fontSize="9" fontFamily="var(--mono)">{p.train_clips}</text></g>)}
        </svg>
      </section>

      <section>
        <h2>The halo effect</h2>
        <p>The five labels move together: raters who like a person score them high on everything. So the five predictions are far from independent.</p>
        <Heat rows={card.trait_correlations.labels} cols={card.trait_correlations.labels} matrix={card.trait_correlations.matrix} rowLabel={(r) => TRAIT_LABEL[r as keyof typeof TRAIT_LABEL] ?? r} />
      </section>

      <section>
        <h2>Emotions vs ratings</h2>
        <p>Correlation between the emotion probabilities and the raw trait scores on the training clips. A smiling face is the strongest single cue.</p>
        <Heat rows={card.emotion_trait_corr.rows} cols={card.emotion_trait_corr.cols} matrix={card.emotion_trait_corr.matrix} rowLabel={(r) => r} />
      </section>

      <section>
        <h2>Limits</h2>
        <ul className="ink2" style={{ maxWidth: 780 }}>
          <li>{d.clips.toLocaleString()} of the 10,000 clips; our own test split, not the official one.</li>
          <li>Labels are first impressions of crowd workers (apparent personality), not a validated questionnaire.</li>
          <li>Demographic bias (age, gender, ethnicity, attractiveness) is likely and was not tested.</li>
          <li>All pretrained networks are frozen; 8 frames per clip; the speech-emotion model was trained on acted speech.</li>
          <li>The app hears Whisper transcripts, training used human transcripts. Synthetic or unusual voices are far outside the training data.</li>
          <li>Never use it to judge, hire, grade or diagnose anyone.</li>
        </ul>
        <p className="tiny muted mono">bundle {card.created_at} · commit {card.git_commit}</p>
      </section>
    </div>
  );
}
