import { describe, expect, it } from "vitest";

import { sampleDisturbanceTorque } from "../wind.js";
import { integrateFull } from "../rigidBody.js";

const I_DEFAULT = [0.075, 0.875, 0.95];

/** Deterministic seedable PRNG (mulberry32) so wind tests are reproducible. */
function seededRng(seed) {
  let a = seed;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("sampleDisturbanceTorque", () => {
  it("returns exactly zero when std is 0", () => {
    const rng = seededRng(1);
    expect(sampleDisturbanceTorque(0, rng)).toEqual([0, 0, 0]);
  });

  it("is a 3-vector with roughly the requested standard deviation", () => {
    const std = 0.05;
    const rng = seededRng(42);
    const samples = [];
    for (let i = 0; i < 2000; i++) {
      samples.push(sampleDisturbanceTorque(std, rng));
    }
    for (let comp = 0; comp < 3; comp++) {
      const vals = samples.map((s) => s[comp]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
      expect(Math.sqrt(variance)).toBeGreaterThan(std * 0.7);
      expect(Math.sqrt(variance)).toBeLessThan(std * 1.3);
    }
  });

  it("is deterministic given the same rng sequence", () => {
    const a = sampleDisturbanceTorque(0.1, seededRng(7));
    const b = sampleDisturbanceTorque(0.1, seededRng(7));
    expect(a).toEqual(b);
  });
});

describe("wind coupling into the integrator", () => {
  it("a constant disturbance torque breaks energy conservation even with no active controller", () => {
    const wind = sampleDisturbanceTorque(0.2, seededRng(3));
    const torqueFn = () => wind;

    const w0 = [0.5, 1.0, 0.2];
    const { omegas } = integrateFull(w0, I_DEFAULT, 5.0, 500, { torqueFn });

    const H = (w) => 0.5 * (w[0] ** 2 * I_DEFAULT[0] + w[1] ** 2 * I_DEFAULT[1] + w[2] ** 2 * I_DEFAULT[2]);
    expect(Math.abs(H(omegas[0]) - H(omegas[omegas.length - 1]))).toBeGreaterThan(1e-4);
  });

  it("wind stacks additively with an active controller's torque", () => {
    const wind = sampleDisturbanceTorque(0.05, seededRng(9));
    const baseTorque = [0.1, -0.2, 0.3];
    const combinedFn = (t, w, q) => [
      baseTorque[0] + wind[0],
      baseTorque[1] + wind[1],
      baseTorque[2] + wind[2],
    ];
    // Sanity check the composition itself is well-formed and finite -- the
    // real controller+wind interaction is exercised end-to-end in
    // scenarioRunner.test.js.
    const w0 = [0, 1.0, 0];
    const { omegas } = integrateFull(w0, I_DEFAULT, 2.0, 200, { torqueFn: combinedFn });
    for (const w of omegas) {
      for (const c of w) expect(Number.isFinite(c)).toBe(true);
    }
  });
});
