import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { validateScenario } from "../core/DataLoader.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

describe("committed sample scenario", () => {
  it("validates against the documented schema", () => {
    const sample = JSON.parse(
      readFileSync(path.join(__dirname, "fixtures/sample_scenario.json"), "utf-8")
    );
    expect(() => validateScenario(sample)).not.toThrow();
  });
});
