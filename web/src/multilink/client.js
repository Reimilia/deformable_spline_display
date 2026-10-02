export class DynamicsClient {
  constructor(onProgress = () => {}) {
    this.pending = new Map();
    this.nextId = 0;
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.worker.onmessage = ({ data }) => {
      if (data.progress) { onProgress(data.progress); return; }
      const job = this.pending.get(data.id);
      if (!job) return;
      this.pending.delete(data.id);
      if (data.error) job.reject(new Error(data.error));
      else job.resolve(data.result);
    };
    this.worker.onerror = event => this.close(new Error(event.message || "WASM worker failed"));
    this.worker.onmessageerror = () => this.close(new Error("Unable to read the dynamics response"));
  }

  request(type, args = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.nextId;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, ...args });
    });
  }

  initialize(base) { return this.request("init", { base }); }
  run(config) { return this.request("run", { config }); }

  close(error = new Error("Dynamics stopped. Reload WASM to run again.")) {
    this.worker.terminate();
    for (const job of this.pending.values()) job.reject(error);
    this.pending.clear();
  }
}
