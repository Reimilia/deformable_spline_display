// Python, NumPy and checkpoint inference stay off the rendering thread.
let ready, pyodide, manifest, modelsURL;
const files = new Set();

async function checkedFetch(url, type = "text") {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Unable to load ${new URL(url).pathname} (${response.status})`);
  return type === "bytes" ? new Uint8Array(await response.arrayBuffer()) : response.text();
}

async function initialize(base) {
  const root = new URL(base);
  modelsURL = new URL("models/", root);
  const runtime = new URL("wasm/0.28.3/", root);
  self.postMessage({ progress: "Loading Python WASM runtime…" });
  const { loadPyodide } = await import(/* @vite-ignore */ new URL("pyodide.mjs", runtime).href);
  pyodide = await loadPyodide({ indexURL: runtime.href });
  self.postMessage({ progress: "Loading NumPy and automatic differentiation…" });
  await pyodide.loadPackage(["numpy", "autograd"]);
  const [catalog, source] = await Promise.all([
    checkedFetch(new URL("manifest.json", modelsURL)),
    checkedFetch(new URL("python/browser_engine.py", root)),
  ]);
  manifest = JSON.parse(catalog);
  if (manifest.schema_version !== 2 || !manifest.checkpoints?.length) throw new Error("Invalid checkpoint manifest");
  pyodide.FS.mkdirTree("/models");
  pyodide.FS.writeFile("/browser_engine.py", source);
  pyodide.globals.set("manifest_json", catalog);
  pyodide.globals.set("report_progress", (kind, done, total) => {
    const message = kind === "checkpoint" ? `Integrating checkpoint ${done}/${total}…` : `Integrating step ${done}/${total}…`;
    self.postMessage({ progress: message });
  });
  await pyodide.runPythonAsync(`
import sys, json
sys.path.insert(0, "/")
from browser_engine import BrowserEngine
engine = BrowserEngine(json.loads(manifest_json), "/models", report_progress)
`);
  return manifest;
}

async function download(entry) {
  if (!entry) throw new Error("Unknown checkpoint");
  for (const [name, sha] of [[entry.weights_file, entry.weights_sha256], [entry.variant_file, null]]) {
    if (files.has(name)) continue;
    self.postMessage({ progress: "Loading checkpoint parameters and saved tasks…" });
    const bytes = await checkedFetch(new URL(name, modelsURL), "bytes");
    if (sha) {
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      const actual = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
      if (actual !== sha) throw new Error("Checkpoint parameter checksum mismatch");
    }
    pyodide.FS.writeFile(`/models/${name}`, bytes);
    files.add(name);
  }
}

let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  // Serialize access to one Python interpreter even if a caller sends twice.
  queue = queue.then(async () => {
    try {
      if (data.type === "init") {
        ready ||= initialize(data.base);
        self.postMessage({ id: data.id, result: await ready });
      } else if (data.type === "run") {
        await ready;
        for (const id of [data.config.checkpoint, data.config.compare].filter(Boolean)) {
          await download(manifest.checkpoints.find(e => e.id === id));
        }
        pyodide.globals.set("request_json", JSON.stringify(data.config));
        const json = await pyodide.runPythonAsync("json.dumps(engine.run(json.loads(request_json)), allow_nan=False, separators=(',', ':'))");
        self.postMessage({ id: data.id, result: JSON.parse(json) });
      } else throw new Error("Unknown dynamics request");
    } catch (error) {
      self.postMessage({ id: data.id, error: error.message });
    }
  });
};
