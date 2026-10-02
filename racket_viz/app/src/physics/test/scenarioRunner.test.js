import { describe, expect, it } from "vitest";

import { buildScenario } from "../scenarioRunner.js";
import { validateScenario } from "../../core/DataLoader.js";

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

const BASE = {
  startAxis: "imid",
  T: 5.0,
  N: 100,
  rng: seededRng(1),
};

describe("buildScenario schema conformance", () => {
  it("background uses the schema's snake_case keys, not buildStaticCurves' camelCase", () => {
    const doc = buildScenario({ ...BASE, controlOn: false, windOn: false });
    expect(doc.background.stable_fixed_points).toBeDefined();
    expect(doc.background.unstable_fixed_points).toBeDefined();
    expect(doc.background.stableFixedPoints).toBeUndefined();
    expect(doc.background.unstableFixedPoints).toBeUndefined();
  });

  it("free, no wind", () => {
    const doc = buildScenario({ ...BASE, controlOn: false, windOn: false });
    expect(() => validateScenario(doc)).not.toThrow();
    expect(doc.meta.mode).toBe("free");
    expect(doc.meta.H_axis).toHaveLength(3);
    expect(doc.meta.wind_on).toBe(false);
  });

  it("R_cas for a free imin/imax start matches |M0| -- the dot sits at its own fixed point, not deep inside the sphere", () => {
    // Regression: R_cas used to always be computed from imid regardless of
    // startAxis. For a free run that's fine when startAxis IS imid (R_cas
    // then approximately equals I[imid]*spinRate = |M0|), but for imin/imax
    // it silently put the whole background (fixed points, trajectory family,
    // and hence the moving dot's apparent position on the normalized sphere)
    // on the wrong scale -- e.g. for this app's default I, I[imin]*spinRate
    // is over 10x smaller than I[imid]*spinRate, so the dot rendered near
    // the sphere's center instead of near the imin fixed point.
    const doc = buildScenario({ ...BASE, startAxis: "imin", startingPerturbation: 0, controlOn: false, windOn: false });
    const M0 = doc.frames.M_body[0];
    expect(Math.hypot(...M0) / doc.meta.R_cas).toBeCloseTo(1, 2);
  });

  it("controlled scenarios base R_cas on the TARGET axis (endAxis), so a successful run's dot lands exactly on the target's marker", () => {
    // R_cas is a pure display scale (desired_L/desired_H, used for actual
    // achieved-detection, are computed from omegaStar/I directly, never from
    // R_cas) -- but it determines where the sphere panel's fixed points and
    // dot land. Basing it on endAxis means |M(t)| -> desired_L ~= R_cas as
    // the controller converges, so the dot visibly reaches the target's own
    // green/red marker instead of some unrelated imid-based radius.
    const doc = buildScenario({ ...BASE, startAxis: "imin", endAxis: "imax", controlOn: true, windOn: false });
    const I = doc.meta.I;
    const expectedRcasApprox = I[2] * (2 * Math.PI); // imax index 2, DEFAULT_SPIN_RATE
    expect(Math.abs(doc.meta.R_cas - expectedRcasApprox) / expectedRcasApprox).toBeLessThan(0.01);

    const Mlast = doc.frames.M_body[doc.frames.M_body.length - 1];
    expect(Math.hypot(...Mlast) / doc.meta.R_cas).toBeCloseTo(1, 1);
  });

  it("startingPerturbation now generalizes to imin/imax starts, not just imid", () => {
    // Regression: the old hardcoded kick only ever applied when
    // startAxis===imid (and only with control/wind both off). A free-spin
    // start on a STABLE axis used to sit at an exact, unwobbling fixed point
    // forever; with a nonzero startingPerturbation it should now trace a
    // small near-axis orbit instead (M_body visibly changing over time).
    const doc = buildScenario({ ...BASE, startAxis: "imin", startingPerturbation: 0.02, controlOn: false, windOn: false });
    const M0 = doc.frames.M_body[0];
    const Mlast = doc.frames.M_body[doc.frames.M_body.length - 1];
    const drift = Math.hypot(...M0.map((v, i) => v - Mlast[i]));
    expect(drift).toBeGreaterThan(1e-3);
  });

  it("startingPerturbation=0 reproduces an exact on-axis start with no wobble, for a stable axis", () => {
    const doc = buildScenario({ ...BASE, startAxis: "imin", startingPerturbation: 0, controlOn: false, windOn: false });
    const M0 = doc.frames.M_body[0];
    for (const M of doc.frames.M_body) {
      expect(Math.hypot(...M.map((v, i) => v - M0[i]))).toBeLessThan(1e-6);
    }
  });

  it("controller_torque is all zero when control is off, even with wind on", () => {
    const doc = buildScenario({ ...BASE, controlOn: false, windOn: true, windStd: 0.05 });
    for (const tau of doc.frames.controller_torque) {
      expect(tau).toEqual([0, 0, 0]);
    }
  });

  it("controlled, no wind", () => {
    const doc = buildScenario({ ...BASE, controlOn: true, endAxis: "imax", windOn: false });
    expect(() => validateScenario(doc)).not.toThrow();
    expect(doc.meta.mode).toBe("controlled");
    expect(typeof doc.meta.desired_H).toBe("number");
    expect(typeof doc.meta.desired_L).toBe("number");
    expect(doc.meta.wind_on).toBe(false);
    // Non-trivial (the controller is doing real work correcting toward the target).
    const anyNonZero = doc.frames.controller_torque.some((tau) => tau.some((x) => x !== 0));
    expect(anyNonZero).toBe(true);
  });

  it("free, with wind", () => {
    const doc = buildScenario({ ...BASE, controlOn: false, windOn: true, windStd: 0.05 });
    expect(() => validateScenario(doc)).not.toThrow();
    expect(doc.meta.mode).toBe("free");
  });

  it("controlled, with wind", () => {
    const doc = buildScenario({
      ...BASE,
      controlOn: true,
      endAxis: "imin",
      windOn: true,
      windStd: 0.05,
    });
    expect(() => validateScenario(doc)).not.toThrow();
    expect(doc.meta.mode).toBe("controlled");
  });

  it("a non-default geometry/mass combination still produces a valid, converging scenario", () => {
    const doc = buildScenario({
      ...BASE,
      handleLength: 1.4,
      hoopLength: 1.6,
      hoopWidth: 0.9,
      totalMass: 1.7,
      hoopFraction: 0.45,
      controlOn: true,
      endAxis: "imax",
      windOn: false,
      T: 10.0,
      N: 500,
    });
    expect(() => validateScenario(doc)).not.toThrow();

    const finalM = doc.frames.M_body[doc.frames.M_body.length - 1];
    const desiredL = doc.meta.desired_L;
    const finalMag = Math.sqrt(finalM[0] ** 2 + finalM[1] ** 2 + finalM[2] ** 2);
    expect(Math.abs(finalMag - desiredL) / desiredL).toBeLessThan(0.1);
  });

  it("frames arrays all have consistent, non-trivial length", () => {
    const doc = buildScenario({ ...BASE, controlOn: false, windOn: false, N: 250 });
    expect(doc.frames.t).toHaveLength(250);
    expect(doc.frames.quaternion).toHaveLength(250);
    expect(doc.frames.M_body).toHaveLength(250);
  });
});
