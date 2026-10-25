// Small, data-only chart pieces shared by the lanes.

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
export const fmt = (x: number | null | undefined, digits = 2) => (x === null || x === undefined ? "–" : x.toFixed(digits));
export const ordinal = (n: number) => {
  const v = Math.round(n);
  const s = ["th", "st", "nd", "rd"];
  const m = v % 100;
  return v + (s[(m - 20) % 10] || s[m] || s[0]);
};
export const clock = (t: number) => `${Math.floor(t / 60)}:${Math.floor(t % 60).toString().padStart(2, "0")}`;

/** Horizontal probability bars (all classes, model order), the largest one highlighted. */
export function ProbBars({ labels, values }: { labels: string[]; values: number[] }) {
  const top = values.indexOf(Math.max(...values));
  return (
    <div className="bars">
      {labels.map((label, i) => (
        <div key={label} className={`bar-row ${i === top ? "top" : ""}`}>
          <span className={i === top ? "" : "ink2"}>{label}</span>
          <span className="track"><i style={{ width: pct(values[i] ?? 0, 1) }} /></span>
          <span>{pct(values[i] ?? 0)}</span>
        </div>
      ))}
    </div>
  );
}

/** SVG path through a series with gaps (null = no value, e.g. an unvoiced frame). */
export function seriesPath(values: (number | null)[], w: number, h: number, lo: number, hi: number) {
  let d = "";
  let pen = false;
  const n = values.length;
  values.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) { pen = false; return; }
    const x = n > 1 ? (i / (n - 1)) * w : 0;
    const y = h - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo || 1)) * h;
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  });
  return d;
}
