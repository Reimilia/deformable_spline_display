import { describe, expect, it } from "vitest";

import {
  computeH,
  referencesForScenario,
  createHamiltonianScene,
  updateEnergyEllipsoid,
  computeAutoFitDistance,
} from "../scenes/ControlPanel.js";

describe("computeH", () => {
  it("computes H = 0.5 * sum(M_i^2 / I_i)", () => {
    const M_body = [[1.5, 0.0, 0.0]];
    const I = [0.075, 0.875, 0.95];
    const [H] = computeH(M_body, I);
    expect(H).toBeCloseTo(0.5 * (1.5 * 1.5) / 0.075, 9);
  });

  it("is conserved for a constant-M free trajectory (regression against panel-3 style series)", () => {
    const M_body = [
      [1.5, 0, 0],
      [1.5, 0, 0],
      [1.5, 0, 0],
    ];
    const I = [0.075, 0.875, 0.95];
    const H = computeH(M_body, I);
    expect(H[0]).toBeCloseTo(H[1], 9);
    expect(H[1]).toBeCloseTo(H[2], 9);
  });
});

describe("referencesForScenario", () => {
  it("returns 3 references (imin/sep/imax) for a free scenario", () => {
    const refs = referencesForScenario({ mode: "free", H_axis: [11.5, 1.0, 0.9] });
    expect(refs).toHaveLength(3);
    expect(refs.map((r) => r.value)).toEqual([11.5, 1.0, 0.9]);
    // colors must be distinct
    expect(new Set(refs.map((r) => r.color)).size).toBe(3);
  });

  it("returns exactly 1 reference (desired_H) for a controlled scenario", () => {
    const refs = referencesForScenario({ mode: "controlled", desired_H: 18.75 });
    expect(refs).toHaveLength(1);
    expect(refs[0].value).toBe(18.75);
  });
});

describe("createHamiltonianScene", () => {
  it("returns a group with a reference sphere, both ellipsoids, and a dot, with desiredEllipsoid hidden by default", () => {
    const { group, currentEllipsoid, desiredEllipsoid, dot } = createHamiltonianScene();
    expect(group.children.length).toBeGreaterThanOrEqual(4);
    expect(desiredEllipsoid.visible).toBe(false);
    expect(currentEllipsoid).toBeDefined();
    expect(dot).toBeDefined();
  });
});

describe("updateEnergyEllipsoid", () => {
  it("scales each axis to sqrt(2*H*I_k)/R_cas", () => {
    const { currentEllipsoid } = createHamiltonianScene();
    const I = [0.075, 0.875, 0.95];
    const H = 2.0;
    const R_cas = 1.5;
    updateEnergyEllipsoid(currentEllipsoid, H, I, R_cas);
    expect(currentEllipsoid.scale.x).toBeCloseTo(Math.sqrt(2 * H * I[0]) / R_cas, 9);
    expect(currentEllipsoid.scale.y).toBeCloseTo(Math.sqrt(2 * H * I[1]) / R_cas, 9);
    expect(currentEllipsoid.scale.z).toBeCloseTo(Math.sqrt(2 * H * I[2]) / R_cas, 9);
  });

  it("clamps to 0 instead of NaN for a negative H (shouldn't happen physically, but stay defensive)", () => {
    const { currentEllipsoid } = createHamiltonianScene();
    updateEnergyEllipsoid(currentEllipsoid, -1, [0.075, 0.875, 0.95], 1.5);
    expect(currentEllipsoid.scale.x).toBe(0);
  });
});

describe("computeAutoFitDistance", () => {
  it("reproduces the original hardcoded camera distance (~3.17) for the common maxExtent=1 case", () => {
    // Regression: the fix must not visibly change framing for the vast
    // majority of scenarios where nothing is oversized.
    const distance = computeAutoFitDistance(1, 45);
    expect(distance).toBeCloseTo(3.169, 1);
  });

  it("scales linearly with maxExtent, so a 6x oversized ellipsoid gets ~6x the distance", () => {
    const d1 = computeAutoFitDistance(1, 45);
    const d6 = computeAutoFitDistance(6, 45);
    expect(d6 / d1).toBeCloseTo(6, 6);
  });

  it("never returns 0 or negative for a degenerate (0 or negative) maxExtent", () => {
    expect(computeAutoFitDistance(0, 45)).toBeGreaterThan(0);
    expect(computeAutoFitDistance(-5, 45)).toBeGreaterThan(0);
  });
});
