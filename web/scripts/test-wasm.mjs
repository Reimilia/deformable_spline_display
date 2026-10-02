// Run the evaluation code in actual WASM, against PyTorch regression fixtures.
import { loadPyodide } from "pyodide";
import { readFile, readdir, mkdir, writeFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

const root = fileURLToPath(new URL("../../", import.meta.url));
const runtime = `${root}web/public/wasm/0.28.3/`;
const pyodide = await loadPyodide({ indexURL: runtime });
await pyodide.loadPackage(["numpy", "autograd"]);
pyodide.FS.mkdirTree("/models");
for (const file of await readdir(`${root}web/public/models`)) {
  pyodide.FS.writeFile(`/models/${file}`, await readFile(`${root}web/public/models/${file}`));
}
pyodide.FS.writeFile("/browser_engine.py", await readFile(`${root}web/public/python/browser_engine.py`));
pyodide.runPython(`
import sys, json
sys.path.insert(0, "/")
from browser_engine import BrowserEngine
engine = BrowserEngine(json.load(open('/models/manifest.json')), '/models')
`);
const cases = JSON.parse(gunzipSync(await readFile(`${root}validation/wasm-cases.json.gz`)));
let maximum = 0;
function close(actual, expected, path = "") {
  if (typeof expected === "number") {
    assert.equal(typeof actual, "number", path);
    assert.ok(Number.isFinite(actual), path);
    const error = Math.abs(actual - expected);
    maximum = Math.max(maximum, error);
    assert.ok(error <= 2e-9 + 2e-8 * Math.abs(expected), `${path}: ${actual} vs ${expected}`);
  } else if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, path);
    expected.forEach((value, i) => close(actual[i], value, `${path}[${i}]`));
  }
}
for (const [index, test] of cases.entries()) {
  pyodide.globals.set("request_json", JSON.stringify(test.config));
  const actual = JSON.parse(pyodide.runPython("json.dumps(engine.run(json.loads(request_json)), allow_nan=False)"));
  assert.equal(actual.runs.length, test.expected.runs.length);
  test.expected.runs.forEach((run, i) => {
    for (const key of ["body", "world", "poses", "phase_q", "phase_p", "target_body", "target_world", "reference_body", "reference_world"]) {
      close(actual.runs[i][key], run[key], key);
    }
    for (const key of Object.keys(run.metrics)) close(actual.runs[i].metrics[key], run.metrics[key], key);
  });
  console.log(`WASM PASS ${index + 1}/${cases.length}: ${test.config.checkpoint}`);
}
await mkdir(`${root}validation`, { recursive: true });
const report = { cases: cases.length, runtime: pyodide.version, maximum_absolute_error: maximum, all_passed: true };
await writeFile(`${root}validation/wasm-report.json`, JSON.stringify(report, null, 2) + "\n");
console.log(report);
