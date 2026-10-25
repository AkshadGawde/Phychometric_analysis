import { MODALITIES, type FusionData, type Modality, type TestMetrics, type Trait } from "../types";

// The same maths as src/predict.py (Engine._fusion), so the ablation toggles in the browser
// give exactly the numbers the server would give with that modality removed.

export const subsetKey = (mods: Iterable<Modality>) => {
  const set = new Set(mods);
  return MODALITIES.filter((m) => set.has(m)).join("+");
};

/** Position of `value` among the held-out clips (101 stored quantiles), in %. */
export function percentile(value: number, quantiles: number[]): number {
  const n = quantiles.length;
  if (value <= quantiles[0]) return 0;
  if (value >= quantiles[n - 1]) return 100;
  for (let i = 1; i < n; i++) {
    if (value <= quantiles[i]) {
      const lo = quantiles[i - 1], hi = quantiles[i];
      const frac = hi > lo ? (value - lo) / (hi - lo) : 0;
      return ((i - 1 + frac) / (n - 1)) * 100;
    }
  }
  return 100;
}

export interface Refused {
  prob: number; percentile: number; subset: string;
  weights: Partial<Record<Modality, number>>; test: TestMetrics;
}

/** Fuse one trait from a chosen subset of the available modalities. */
export function refuse(fusion: FusionData, trait: Trait, subset: string): Refused | null {
  const option = fusion.options[subset]?.[trait];
  if (!option) return null;
  let prob = 0;
  for (const [m, w] of Object.entries(option.weights) as [Modality, number][]) {
    const p = fusion.unimodal_prob[m]?.[trait];
    if (p === undefined) return null;
    prob += w * p;
  }
  return { prob, percentile: percentile(prob, option.quantiles), subset, weights: option.weights, test: option.test };
}
