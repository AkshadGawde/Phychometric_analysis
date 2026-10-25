// End-to-end check of the running app (ROADMAP_WEB.md, Phase 7), with headless Chromium.
//   node scripts/e2e.mjs [path/to/video-or-audio]  [screenshot-dir]
// 1. replays the first demo sample and waits for the result;
// 2. if a file is given, uploads it like a user would and waits for the result.
// Needs the API (port 8000) and Vite (port 5173), or the built app on BASE_URL.
import { chromium } from "@playwright/test";

const base = process.env.BASE_URL ?? "http://localhost:5173";
const [file, shots] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
const fail = async (msg) => { console.error("E2E FAILED: " + msg); await browser.close(); process.exit(1); };

// 1. demo replay
await page.goto(base + "/");
await page.getByRole("tab", { name: /Demo samples/ }).click();
await page.locator(".sample").first().click();
await page.waitForSelector(".result", { timeout: 90000 }).catch(() => fail("replay: no result"));
const traits = await page.locator(".result .trait").count();
if (traits !== 5) await fail(`replay: ${traits} traits shown`);
// toggling a lane must change the fusion choice text
await page.locator(".toggle", { hasText: "Words" }).click();
const rule = await page.locator(".toggles .tiny").innerText();
if (!rule.includes("fusing Voice")) await fail(`replay: ablation toggle did not apply (${rule})`);
console.log("1. replay: 5 traits, ablation toggle works");

// 2. live upload
if (file) {
  await page.getByRole("button", { name: "New input" }).click();
  await page.getByRole("tab", { name: /Upload a file/ }).click();
  await page.locator(".consent input").check();
  const t0 = Date.now();
  await page.locator("input[type=file]").setInputFiles(file);
  await page.waitForSelector(".result", { timeout: 240000 }).catch(() => fail("upload: no result"));
  const frames = await page.locator(".frame").count();
  const words = await page.locator(".tok").count();
  console.log(`2. upload: result after ${((Date.now() - t0) / 1000).toFixed(1)} s · ${frames} face frames · ${words} words`);
  if (shots) {
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${shots}/upload-done.png`, fullPage: true });
  }
}
if (errors.length) await fail("browser errors: " + errors.join(" | "));
console.log("E2E PASSED");
await browser.close();
