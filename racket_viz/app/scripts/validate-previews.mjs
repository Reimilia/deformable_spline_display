import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { validateRollout } from "../src/multilink/plots.js";
import { combinePreviews } from "../src/multilink/source.js";

const folder = path.resolve("../data/previews");
const manifest = JSON.parse(fs.readFileSync(path.join(folder, "manifest.json"), "utf8"));
assert.equal(manifest.schema_version, 1);
assert.equal(manifest.catalog.count, manifest.catalog.checkpoints.length);
assert(manifest.previews.length >= 2 * manifest.catalog.count);
const used = new Set(), reference = new Map();
for (const e of manifest.catalog.checkpoints) {
  assert(!/measurement/i.test(e.label));
  assert(e.comparison_group);
  for (const [group, count] of Object.entries(e.tasks)) {
    for (let index = 0; index < count; index++) {
      const p = manifest.previews.find(p => p.checkpoint === e.id && p.group === group && p.task_index === index);
      assert(p, `Missing preview for ${e.id} ${group} ${index}`);
      assert(/^[a-f0-9]{20}\.json$/.test(p.file));
      assert(!used.has(p.file));used.add(p.file);
      const result = validateRollout(JSON.parse(fs.readFileSync(path.join(folder, p.file), "utf8")));
      assert.equal(result.runs[0].id, e.id);
      assert.equal(result.group, group);
      assert.equal(result.task_index, index);
      const key = `${e.comparison_group}:${group}:${index}`;
      if (reference.has(key)) combinePreviews(reference.get(key), result);
      else reference.set(key, result);
    }
  }
}
assert.equal(used.size, manifest.previews.length);
console.log(`Validated ${used.size} previews across ${manifest.catalog.count} checkpoints, including stage comparisons.`);
