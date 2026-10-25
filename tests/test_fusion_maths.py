"""Phase 5 acceptance: the ablation toggles in the browser give the same numbers as Python.

For every recorded replay (app/replays/*.json) and every subset of its available lanes, the
browser's fusion code (web/src/lib/fusion.ts, run with node) must match the server:
  - for the default choice per trait: the fused probability and percentile in the fusion event;
  - for every other subset: the same late-fusion formula and percentile as src/predict.py.

Run:  python tests/test_fusion_maths.py          (needs node + `npm install` in web/)
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import predict  # noqa: E402
import config  # noqa: E402

PROB_TOL = 1e-3          # the event carries 4-decimal probabilities
PCT_TOL = 0.5            # percentile points


def main():
    replays = sorted(glob.glob(os.path.join(config.REPLAYS_DIR, "*.json")))
    if not replays:
        sys.exit("no replays in app/replays - record one with src/predict.py --record NAME")
    failures, checked = [], 0
    for path in replays:
        replay = json.load(open(path))
        fusion = next(e for e in replay["events"] if e["type"] == "stage" and e["stage"] == "fusion")["data"]
        js = json.loads(subprocess.run(["node", "scripts/fusion-cli.mjs", path], cwd=os.path.join(ROOT, "web"),
                                       capture_output=True, text=True, check=True).stdout)
        for subset, per_trait in fusion["options"].items():
            for trait, option in per_trait.items():
                prob = sum(w * fusion["unimodal_prob"][m][trait] for m, w in option["weights"].items())
                expected = {"prob": prob, "percentile": predict.percentile(prob, option["quantiles"])}
                if fusion["traits"][trait]["subset"] == subset:             # the server's own result
                    expected = {"prob": fusion["traits"][trait]["prob"], "percentile": fusion["traits"][trait]["percentile"]}
                got = js[subset][trait]
                checked += 1
                if abs(got["prob"] - expected["prob"]) > PROB_TOL or abs(got["percentile"] - expected["percentile"]) > PCT_TOL:
                    failures.append(f"{os.path.basename(path)} {subset} {trait}: browser {got} vs {expected}")
    if failures:
        print("FUSION MATHS DIFFER:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print(f"FUSION MATHS PASSED: {checked} (replay, subset, trait) combinations agree between browser and Python")


def test_fusion_maths():   # pytest entry point
    main()


if __name__ == "__main__":
    main()
