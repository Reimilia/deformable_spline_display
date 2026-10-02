import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import assert from "node:assert/strict";

const root = fileURLToPath(new URL("../../", import.meta.url));
const dist = path.join(root, "web/dist");
const prefix = (process.env.TEST_BASE_PATH || "/deformation-demo/").replace(/\/+$/, "") + "/";
const mime = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".wasm": "application/wasm", ".json": "application/json" };
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    if (!url.pathname.startsWith(prefix)) throw new Error("outside deployment prefix");
    const file = path.resolve(dist, url.pathname.slice(prefix.length) || "index.html");
    if (!file.startsWith(dist + path.sep)) throw new Error("outside dist");
    const data = await readFile(file);
    res.writeHead(200, { "Content-Type": mime[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  } catch { res.writeHead(404); res.end("Not found"); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {}), args: ["--no-sandbox"] });
const errors = [], forbidden = [], badRequests = [], requests = [];
try {
  const page = await browser.newPage({ viewport: { width: 1536, height: 1080 }, acceptDownloads: true });
  page.on("pageerror", e => errors.push(e.message));
  page.on("request", req => {
    requests.push(req.url());
    if (!req.url().startsWith(origin) || req.url().includes("/api/") || req.url().includes("previews/")) forbidden.push(req.url());
  });
  page.on("response", res => { if (res.status() >= 400) badRequests.push(res.url()); });
  await page.goto(origin + prefix);
  async function finished() {
    await page.waitForFunction(() => {
      const status = document.getElementById("status");
      return /integrated in WASM|Unable to run|WASM startup failed/.test(status.textContent);
    }, null, { timeout: 120000 });
    assert.match(await page.locator("#status").textContent(), /integrated in WASM/);
  }
  async function exported() {
    const download = page.waitForEvent("download");
    await page.locator("#export").click();
    return JSON.parse(await readFile(await (await download).path(), "utf8"));
  }
  await finished();
  assert.equal(await page.locator("#model-count").textContent(), "58 checkpoints");
  assert.equal(await page.locator("#damping").isEnabled(), true);
  assert.equal(await page.locator("#checkpoint option").count(), 18);
  const first = await exported();
  assert.equal(first.runtime, "pyodide-numpy-wasm-float64");
  await page.locator("#play").click();
  await page.locator("#scrub").fill("1.4");
  await mkdir(path.join(root, "validation"), { recursive: true });
  await page.screenshot({ path: path.join(root, "validation/wasm-desktop.png"), fullPage: true });
  const compare = await page.locator("#compare option").evaluateAll(options => options.find(o => /Holonomic.*Final/.test(o.textContent)).value);
  await page.locator("#compare").selectOption(compare);
  await page.locator("#target-angle").fill("25");
  await page.locator("#damping").fill("1.6");
  await page.locator("#run").click(); await finished();
  const second = await exported();
  assert.equal(second.runs.length, 2);
  assert.notDeepEqual(second.runs[0].phase_q, first.runs[0].phase_q);
  assert.equal(second.configuration.target_angle, 25);
  assert.equal(second.configuration.damping, 1.6);
  assert.match(await page.locator("#shape-legend").textContent(), /FIM/);
  assert.doesNotMatch(await page.locator("#shape-legend").textContent(), /measurement/i);
  await page.locator("#family").selectOption("spline");
  assert.equal(await page.locator("#checkpoint option").count(), 40);
  await page.locator("#horizon").fill("1");
  await page.locator("#noise").fill("0.01");
  await page.locator("#group").selectOption("hard_test");
  await page.locator("#run").click(); await finished();
  const spline = await exported();
  assert.equal(spline.family, "spline"); assert.equal(spline.configuration.noise, .01);
  assert.equal(spline.group, "hard_test"); assert.equal(spline.runs[0].body[0].length, 200);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(root, "validation/wasm-mobile.png"), fullPage: true });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  // Cancellation destroys the worker; retry initializes a fresh interpreter.
  await page.locator("#horizon").fill("6");
  await page.locator("#run").click(); await page.locator("#cancel").click();
  await page.waitForFunction(() => !document.getElementById("reload-wasm").disabled);
  await page.locator("#reload-wasm").click(); await finished();
  assert.deepEqual(errors, []); assert.deepEqual(forbidden, []); assert.deepEqual(badRequests, []);
  assert.ok(requests.some(url => url.endsWith(".wasm")));
  const report = { all_passed: true, checkpoints: 58, deployment_prefix: prefix, api_requests: 0, external_requests: 0,
    changed_configuration: true, comparison: true, hard_ood: true, observation_noise: true, cancellation_and_retry: true,
    desktop_and_mobile: true, requests: requests.length };
  await writeFile(path.join(root, "validation/browser-report.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(report);
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
