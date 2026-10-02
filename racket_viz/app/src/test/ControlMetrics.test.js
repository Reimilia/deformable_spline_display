import { describe, expect, it } from "vitest";

import {
  isAchievable,
  computeCumulativeWork,
  computeTorqueMagnitude,
  computeAchievedIndex,
} from "../scenes/ControlMetrics.js";
import { computeH } from "../scenes/ControlPanel.js";
import { buildScenario } from "../physics/scenarioRunner.js";

describe("isAchievable", () => {
  it("is true only for controlled, wind-off, stable-target scenarios", () => {
    expect(isAchievable({ mode: "controlled", wind_on: false, target_axis: 0 })).toBe(true);
    expect(isAchievable({ mode: "controlled", wind_on: false, target_axis: 2 })).toBe(true);
  });

  it("is false with wind on", () => {
    expect(isAchievable({ mode: "controlled", wind_on: true, target_axis: 0 })).toBe(false);
  });

  it("is false targeting the unstable imid axis", () => {
    expect(isAchievable({ mode: "controlled", wind_on: false, target_axis: 1 })).toBe(false);
  });

  it("is false for free scenarios", () => {
    expect(isAchievable({ mode: "free", wind_on: false, target_axis: 0 })).toBe(false);
  });
});

describe("computeCumulativeWork", () => {
  it("integrates torque.omega with the rectangle rule, starting at 0", () => {
    const t = [0, 1, 2];
    const I = [1, 1, 1]; // omega == M_body directly, for a simple hand check
    const M_body = [
      [0, 0, 0],
      [1, 0, 0],
      [1, 0, 0],
    ];
    const torque = [
      [0, 0, 0],
      [2, 0, 0],
      [2, 0, 0],
    ];
    const work = computeCumulativeWork(torque, M_body, I, t);
    expect(work[0]).toBe(0);
    // power = [0, 2, 2]; trapezoidal: work[1] = 0.5*(0+2)*1 = 1
    expect(work[1]).toBeCloseTo(1, 9);
    // work[2] = work[1] + 0.5*(2+2)*1 = 1 + 2 = 3
    expect(work[2]).toBeCloseTo(3, 9);
  });

  it("matches H(t) - H(0) exactly for a real no-wind controlled run (dH/dt = torque.omega)", () => {
    const doc = buildScenario({
      startAxis: "imid",
      controlOn: true,
      endAxis: "imax",
      windOn: false,
      T: 6.0,
      N: 600,
    });
    const H = computeH(doc.frames.M_body, doc.meta.I);
    const work = computeCumulativeWork(doc.frames.controller_torque, doc.frames.M_body, doc.meta.I, doc.frames.t);

    const lastIdx = work.length - 1;
    expect(work[lastIdx]).toBeCloseTo(H[lastIdx] - H[0], 1);
  });
});

describe("computeTorqueMagnitude", () => {
  it("is the Euclidean norm of each torque vector", () => {
    expect(computeTorqueMagnitude([[3, 4, 0], [0, 0, 0]])).toEqual([5, 0]);
  });
});

describe("computeAchievedIndex", () => {
  const baseMeta = { mode: "controlled", wind_on: false, target_axis: 2, desired_L: 1.0 };

  it("returns null when not achievable at all", () => {
    const t = [0, 1, 2];
    const M_body = [[0, 0, 1], [0, 0, 1], [0, 0, 1]];
    expect(computeAchievedIndex(M_body, t, { ...baseMeta, wind_on: true })).toBeNull();
  });

  it("returns null if it never sustains within tolerance", () => {
    const t = [0, 1, 2];
    const M_body = [[1, 0, 0], [0, 1, 0], [1, 0, 0]]; // never near [0,0,1]
    expect(computeAchievedIndex(M_body, t, baseMeta)).toBeNull();
  });

  it("ignores a momentary pass-through that doesn't sustain", () => {
    const t = [0, 1, 2, 3, 4];
    const M_body = [
      [1, 0, 0],
      [0, 0, 1], // momentarily within tolerance...
      [1, 0, 0], // ...but immediately leaves again
      [1, 0, 0],
      [1, 0, 0],
    ];
    expect(computeAchievedIndex(M_body, t, baseMeta, 0.05, 1.0)).toBeNull();
  });

  it("finds the index where sustained convergence begins", () => {
    const t = [0, 1, 2, 3, 4];
    const M_body = [
      [1, 0, 0],
      [1, 0, 0],
      [0, 0, 1], // converges here...
      [0, 0, 1], // ...and stays (>= 1s sustained by t=3)
      [0, 0, 1],
    ];
    expect(computeAchievedIndex(M_body, t, baseMeta, 0.05, 1.0)).toBe(2);
  });
});
