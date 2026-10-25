// Runs the browser's fusion maths (src/lib/fusion.ts) on a replay's fusion event and prints
// the result for every subset of lanes as JSON. Used by tests/test_fusion_maths.py.
//   node scripts/fusion-cli.mjs path/to/replay.json
import { build } from "esbuild";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL("..", import.meta.url));
const out = await build({ entryPoints: [here + "src/lib/fusion.ts"], bundle: true, format: "esm", write: false, platform: "node" });
const mod = await import("data:text/javascript;base64," + Buffer.from(out.outputFiles[0].text).toString("base64"));

const replay = JSON.parse(readFileSync(process.argv[2], "utf8"));
const fusion = replay.events.find((e) => e.type === "stage" && e.stage === "fusion").data;
const result = {};
for (const subset of Object.keys(fusion.options)) {
  result[subset] = {};
  for (const trait of Object.keys(fusion.traits)) {
    const r = mod.refuse(fusion, trait, subset);
    result[subset][trait] = r && { prob: r.prob, percentile: r.percentile };
  }
}
console.log(JSON.stringify(result));
