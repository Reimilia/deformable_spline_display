// One data contract for local inference, hosted inference and static Pages.
export function assetUrl(path, base, pageUrl) {
  return new URL(path, new URL(base, pageUrl)).href;
}

export function normalizeApiBase(input, pageUrl) {
  if (!input.trim()) return "";
  const url = new URL(input.trim());
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
  if (!(["https:", "http:"].includes(url.protocol)) || url.username || url.password || url.search || url.hash) {
    throw new Error("Enter a service URL without credentials, query parameters or a fragment.");
  }
  if (new URL(pageUrl).protocol === "https:" && url.protocol !== "https:" && !local) {
    throw new Error("Use HTTPS for a hosted dynamics service.");
  }
  return url.href.replace(/\/+$/, "") + "/";
}

export function apiUrl(path, customBase, base, pageUrl) {
  return customBase ? new URL(`api/${path}`, customBase).href : assetUrl(`api/${path}`, base, pageUrl);
}

export async function fetchJson(url, init = {}) {
  const response = await fetch(url, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).error || message; } catch { /* static 404 may be HTML */ }
    throw new Error(message);
  }
  return response.json();
}

export function previewFor(manifest, config) {
  const item = manifest.previews.find(p => p.checkpoint === config.checkpoint && p.group === config.group && p.task_index === config.task_index);
  if (!item) throw new Error("This saved task is unavailable. Choose another preview or connect a dynamics service.");
  return item;
}

export function comparisonOptions(entries, selected, mode) {
  return entries.filter(e => e.id !== selected.id && e.family === selected.family &&
    (mode !== "preview" || e.comparison_group === selected.comparison_group));
}

export function combinePreviews(a, b) {
  if (a.family !== b.family || a.group !== b.group || a.task_index !== b.task_index) throw new Error("Preview tasks differ");
  const left = a.runs[0], right = b.runs[0];
  // Only identical material initial states, targets and poses can be overlaid.
  for (const key of ["target_body", "target_pose"]) {
    if (JSON.stringify(left[key]) !== JSON.stringify(right[key])) throw new Error("These previews use different saved targets");
  }
  if (JSON.stringify(left.world[0]) !== JSON.stringify(right.world[0])) throw new Error("These previews use different initial states");
  return { ...a, runs: [left, right] };
}
