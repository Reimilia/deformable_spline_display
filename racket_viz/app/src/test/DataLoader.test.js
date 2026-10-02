import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { validateScenario } from "../core/DataLoader.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sample = JSON.parse(
  readFileSync(path.join(__dirname, "fixtures/sample_scenario.json"), "utf-8")
);

describe("validateScenario", () => {
  it("accepts a well-formed scenario", () => {
    expect(() => validateScenario(sample)).not.toThrow();
  });

  it("rejects a scenario missing a required meta key", () => {
    const bad = structuredClone(sample);
    delete bad.meta.R_cas;
    expect(() => validateScenario(bad)).toThrow(/missing required keys.*R_cas/);
  });

  it("rejects mismatched frame array lengths", () => {
    const bad = structuredClone(sample);
    bad.frames.quaternion.pop();
    expect(() => validateScenario(bad)).toThrow(/length mismatch/);
  });

  it("rejects a controlled scenario missing desired_H/desired_L", () => {
    const bad = structuredClone(sample);
    bad.meta.mode = "controlled";
    expect(() => validateScenario(bad)).toThrow(/desired_H\/desired_L/);
  });

  it("rejects a scenario missing geometry", () => {
    const bad = structuredClone(sample);
    delete bad.geometry;
    expect(() => validateScenario(bad)).toThrow(/missing 'geometry'/);
  });

  it("rejects a scenario missing background", () => {
    const bad = structuredClone(sample);
    delete bad.background;
    expect(() => validateScenario(bad)).toThrow(/missing 'background'/);
  });

  it("rejects background fixed points with the wrong camelCase keys", () => {
    // Regression: scenarioRunner.js once returned buildStaticCurves()'s
    // camelCase output (stableFixedPoints) unrenamed, which this schema check
    // didn't catch because it didn't look at `background` at all -- it only
    // failed downstream, deep inside SphereScene's rendering code.
    const bad = structuredClone(sample);
    delete bad.background.stable_fixed_points;
    bad.background.stableFixedPoints = [[1.5, 0, 0]];
    expect(() => validateScenario(bad)).toThrow(/background missing required keys.*stable_fixed_points/);
  });

  it("rejects a free scenario missing H_axis", () => {
    const bad = structuredClone(sample);
    delete bad.meta.H_axis;
    expect(() => validateScenario(bad)).toThrow(/H_axis/);
  });

  it("rejects a controller_torque array of the wrong length", () => {
    const bad = structuredClone(sample);
    bad.frames.controller_torque.pop();
    expect(() => validateScenario(bad)).toThrow(/length mismatch/);
  });

  it("rejects a controller_torque entry that isn't a 3-vector", () => {
    const bad = structuredClone(sample);
    bad.frames.controller_torque[0] = [1, 2];
    expect(() => validateScenario(bad)).toThrow(/controller_torque entry has length/);
  });

  it("accepts a valid controlled scenario with desired_H/desired_L present", () => {
    const good = structuredClone(sample);
    good.meta.mode = "controlled";
    good.meta.desired_H = 1.0;
    good.meta.desired_L = 1.5;
    expect(() => validateScenario(good)).not.toThrow();
  });
});
