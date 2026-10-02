import { describe, expect, it } from "vitest";

import {
  casimirRadius,
  energyAtAxes,
  icFromH,
  buildStaticCurves,
  analyticSeparatrixCurves,
  kickedIC,
  currentAxisOrbitCurve,
  buildBackgroundWithCurrentOrbit,
} from "../phasePortrait.js";

const I = [0.075, 0.875, 0.95];
const IMIN = 0;
const IMID = 1;
const IMAX = 2;

function norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

describe("separatrix energy", () => {
  it("H_axis[imid] equals R_cas^2 / (2*I[imid])", () => {
    const R_cas = casimirRadius(I, IMID, 1.5);
    const H_axis = energyAtAxes(I, R_cas);
    expect(H_axis[IMID]).toBeCloseTo((R_cas * R_cas) / (2 * I[IMID]), 9);
  });
});

describe("icFromH", () => {
  it("round-trips: |M0|=R_cas and recomputed energy equals H_val", () => {
    const R_cas = casimirRadius(I, IMID, 1.5);
    const H_axis = energyAtAxes(I, R_cas);
    const H_sep = H_axis[IMID];

    const alpha = 0.3;
    const H_val = H_axis[IMIN] + alpha * (H_sep - H_axis[IMIN]);
    const Mmid = 0.001 * R_cas;

    const M = icFromH(H_val, Mmid, I, R_cas, IMIN, IMAX, IMID);
    expect(M).not.toBeNull();
    expect(norm(M)).toBeCloseTo(R_cas, 4);

    const H_recomputed = 0.5 * (M[0] ** 2 / I[0] + M[1] ** 2 / I[1] + M[2] ** 2 / I[2]);
    expect(H_recomputed).toBeCloseTo(H_val, 4);
  });

  it("returns null when infeasible", () => {
    const R_cas = casimirRadius(I, IMID, 1.5);
    const M = icFromH(1e6, 0.0, I, R_cas, IMIN, IMAX, IMID);
    expect(M).toBeNull();
  });
});

describe("analyticSeparatrixCurves", () => {
  const R_cas = casimirRadius(I, IMID, 1.5);

  it("returns exactly 4 curves (2 great circles x 2 arcs each)", () => {
    // Regression: an earlier version paired signs incorrectly and produced 2
    // curves that were BOTH on the same great circle (one of them wasn't
    // even a valid trajectory of Euler's equations, just coincidentally on
    // the right Casimir/energy level) -- see the next test.
    const curves = analyticSeparatrixCurves(I, R_cas, IMIN, IMAX, IMID);
    expect(curves).toHaveLength(4);
  });

  it("satisfies Euler's equations to floating-point precision at every point, for all 4 sign combinations (central-difference check)", () => {
    // analyticSeparatrixCurves samples in index space; recompute its own
    // formula directly here (mirroring the function) so we know the exact
    // time value behind each sample and can finite-difference against it.
    const I1 = I[IMIN], I2 = I[IMID], I3 = I[IMAX];
    const lambda = Math.sqrt(R_cas * R_cas * (1 / I1 - 1 / I2) * (1 / I2 - 1 / I3));
    const k = ((1 / I2 - 1 / I3) * R_cas) / lambda;
    const P = (R_cas * lambda * I1 * I3) / (I3 - I1);
    const B = Math.sqrt(P / k);
    const A = k * B;
    const Mfn = (circleSign, arcSign, t) => {
      const sech = 1 / Math.cosh(lambda * t);
      const M = [0, 0, 0];
      M[IMIN] = arcSign * A * sech;
      M[IMID] = circleSign * R_cas * Math.tanh(lambda * t);
      M[IMAX] = circleSign * arcSign * B * sech;
      return M;
    };
    const dMdt = (M) => [
      (1 / I[2] - 1 / I[1]) * M[1] * M[2],
      (1 / I[0] - 1 / I[2]) * M[2] * M[0],
      (1 / I[1] - 1 / I[0]) * M[0] * M[1],
    ];
    const h = 1e-6;
    for (const circleSign of [1, -1]) {
      for (const arcSign of [1, -1]) {
        for (const t of [-5, -1, -0.1, 0, 0.1, 1, 5]) {
          const numeric = [0, 1, 2].map(
            (i) => (Mfn(circleSign, arcSign, t + h)[i] - Mfn(circleSign, arcSign, t - h)[i]) / (2 * h)
          );
          const analytic = dMdt(Mfn(circleSign, arcSign, t));
          const residual = Math.hypot(...numeric.map((v, i) => v - analytic[i]));
          expect(residual).toBeLessThan(1e-6);
        }
      }
    }
  });

  it("(+,-) sign pairing (a real bug caught during implementation) is NOT a valid trajectory, despite sitting on the right level set", () => {
    // Documents why the fix above was necessary: this combination conserves
    // |M| and H exactly (it's a real point on the level set) but fails
    // Euler's equations outright -- |M|/H conservation alone doesn't prove a
    // curve is a genuine trajectory.
    const I1 = I[IMIN], I2 = I[IMID], I3 = I[IMAX];
    const lambda = Math.sqrt(R_cas * R_cas * (1 / I1 - 1 / I2) * (1 / I2 - 1 / I3));
    const k = ((1 / I2 - 1 / I3) * R_cas) / lambda;
    const P = (R_cas * lambda * I1 * I3) / (I3 - I1);
    const B = Math.sqrt(P / k);
    const A = k * B;
    const invalidM = (t) => {
      const sech = 1 / Math.cosh(lambda * t);
      const M = [0, 0, 0];
      M[IMIN] = A * sech; // sign(+A) paired with...
      M[IMID] = R_cas * Math.tanh(lambda * t);
      M[IMAX] = -B * sech; // ...sign(-B): NOT one of the 4 valid combinations
      return M;
    };
    const dMdt = (M) => [
      (1 / I[2] - 1 / I[1]) * M[1] * M[2],
      (1 / I[0] - 1 / I[2]) * M[2] * M[0],
      (1 / I[1] - 1 / I[0]) * M[0] * M[1],
    ];
    const h = 1e-6, t = 0.7;
    const numeric = [0, 1, 2].map((i) => (invalidM(t + h)[i] - invalidM(t - h)[i]) / (2 * h));
    const analytic = dMdt(invalidM(t));
    const residual = Math.hypot(...numeric.map((v, i) => v - analytic[i]));
    expect(residual).toBeGreaterThan(1);
    // Yet it still sits exactly on the Casimir sphere at the separatrix energy:
    const M = invalidM(t);
    expect(norm(M)).toBeCloseTo(R_cas, 6);
  });

  it("conserves |M|=R_cas and H=H_sep exactly at every sample point", () => {
    const H_sep = energyAtAxes(I, R_cas)[IMID];
    for (const curve of analyticSeparatrixCurves(I, R_cas, IMIN, IMAX, IMID)) {
      for (const M of curve) {
        expect(norm(M)).toBeCloseTo(R_cas, 9);
        const H = 0.5 * (M[0] ** 2 / I[0] + M[1] ** 2 / I[1] + M[2] ** 2 / I[2]);
        expect(H).toBeCloseTo(H_sep, 9);
      }
    }
  });

  it("M[imid] is monotonic along each curve -- a clean single pole-to-pole transit, never bouncing back", () => {
    for (const curve of analyticSeparatrixCurves(I, R_cas, IMIN, IMAX, IMID)) {
      const rising = curve[curve.length - 1][IMID] > curve[0][IMID];
      for (let i = 1; i < curve.length; i++) {
        if (rising) expect(curve[i][IMID]).toBeGreaterThanOrEqual(curve[i - 1][IMID]);
        else expect(curve[i][IMID]).toBeLessThanOrEqual(curve[i - 1][IMID]);
      }
      expect(Math.abs(curve[curve.length - 1][IMID])).toBeCloseTo(R_cas, 3);
      expect(Math.abs(curve[0][IMID])).toBeCloseTo(R_cas, 3);
    }
  });

  it("the 4 curves form 2 pairs by ratio M[imin]/M[imax] = +r and -r (the 2 great circles)", () => {
    const curves = analyticSeparatrixCurves(I, R_cas, IMIN, IMAX, IMID);
    const ratios = curves.map((c) => {
      const mid = Math.floor(c.length / 2);
      return c[mid][IMIN] / c[mid][IMAX];
    });
    const positive = ratios.filter((r) => r > 0);
    const negative = ratios.filter((r) => r < 0);
    expect(positive).toHaveLength(2);
    expect(negative).toHaveLength(2);
    expect(positive[0]).toBeCloseTo(positive[1], 9);
    expect(negative[0]).toBeCloseTo(negative[1], 9);
    expect(positive[0]).toBeCloseTo(-negative[0], 9);
  });

  it("returns [] for a degenerate inertia tensor (no distinct unstable axis)", () => {
    const curves = analyticSeparatrixCurves([0.5, 0.5, 0.9], R_cas, IMIN, IMAX, IMID);
    expect(curves).toEqual([]);
  });
});

describe("kickedIC", () => {
  const spinRate = 2 * Math.PI;

  it("kickFraction=0 reproduces an exact on-axis IC, for any start axis", () => {
    for (const axis of [IMIN, IMID, IMAX]) {
      const M0 = kickedIC(I, IMIN, IMID, IMAX, axis, spinRate, 0);
      const expected = [0, 0, 0];
      expected[axis] = I[axis] * spinRate;
      for (let i = 0; i < 3; i++) expect(M0[i]).toBeCloseTo(expected[i], 9);
    }
  });

  it("prefers imin as the kick axis, falling back to imid when starting on imin itself", () => {
    const M0mid = kickedIC(I, IMIN, IMID, IMAX, IMID, spinRate, 0.02);
    expect(M0mid[IMIN]).not.toBeCloseTo(0, 6); // kick landed on imin
    const M0min = kickedIC(I, IMIN, IMID, IMAX, IMIN, spinRate, 0.02);
    expect(M0min[IMID]).not.toBeCloseTo(0, 6); // kick landed on imid (imin is the start axis here)
  });

  it("|M0| stays close to the start axis's own baseline (I[axis]*spinRate), not R_cas", () => {
    // Regression: an earlier draft scaled the kick by R_cas uniformly, which
    // for imin/imax silently changes the baseline spin rate, since R_cas is
    // built from I[imid] specifically and can differ from I[imin]*spinRate
    // by an order of magnitude.
    for (const axis of [IMIN, IMID, IMAX]) {
      const baseline = I[axis] * spinRate;
      const M0 = kickedIC(I, IMIN, IMID, IMAX, axis, spinRate, 0.02);
      const mag = Math.hypot(...M0);
      expect(Math.abs(mag - baseline) / baseline).toBeLessThan(0.01);
    }
  });

  it("larger kickFraction moves further off-axis", () => {
    const M0small = kickedIC(I, IMIN, IMID, IMAX, IMID, spinRate, 0.01);
    const M0large = kickedIC(I, IMIN, IMID, IMAX, IMID, spinRate, 0.05);
    expect(Math.abs(M0large[IMIN])).toBeGreaterThan(Math.abs(M0small[IMIN]));
  });
});

describe("buildStaticCurves", () => {
  it("draws exactly 4 trajectory curves (one per stable fixed point), not 8 redundant duplicates", () => {
    // Regression: familyCurves() used to loop over both sa AND sb signs (4
    // combinations per stable axis), but (sa, sb=+1) and (sa, sb=-1) trace
    // the SAME closed orbit from two different starting phases -- drawing
    // both wastefully doubled every loop and, since each is independently
    // integrated, the two near-duplicate copies don't perfectly coincide,
    // rendering as a visibly thicker line than a single clean pass.
    const R_cas = casimirRadius(I, IMID, 1.5);
    const curves = buildStaticCurves(I, R_cas, IMIN, IMAX, IMID);
    expect(curves.trajectories).toHaveLength(4);
  });

  it("every background curve and fixed point stays on the Casimir sphere", () => {
    const R_cas = casimirRadius(I, IMID, 1.5);
    const curves = buildStaticCurves(I, R_cas, IMIN, IMAX, IMID);

    for (const curve of [...curves.trajectories, ...curves.separatrices]) {
      for (const M of curve) {
        expect(norm(M) / R_cas).toBeCloseTo(1, 2);
      }
    }
    for (const fp of [...curves.stableFixedPoints, ...curves.unstableFixedPoints]) {
      expect(norm(fp)).toBeCloseTo(R_cas, 5);
    }
  });
});

describe("currentAxisOrbitCurve", () => {
  const R_cas = casimirRadius(I, IMID, 1.5);
  const spinRate = 2 * Math.PI;

  it("returns null when kickFraction is 0 (nothing to show, exact on-axis start doesn't move)", () => {
    expect(currentAxisOrbitCurve(I, R_cas, IMIN, IMID, IMAX, IMID, spinRate, 0)).toBeNull();
  });

  it("starts at exactly the same M0 kickedIC would use for the real run", () => {
    const expected = kickedIC(I, IMIN, IMID, IMAX, IMID, spinRate, 0.02);
    const curve = currentAxisOrbitCurve(I, R_cas, IMIN, IMID, IMAX, IMID, spinRate, 0.02);
    for (let i = 0; i < 3; i++) expect(curve[0][i]).toBeCloseTo(expected[i], 9);
  });

  it("for an imid start, shows a real transit -- M[imid] changes sign within the returned duration", () => {
    const curve = currentAxisOrbitCurve(I, R_cas, IMIN, IMID, IMAX, IMID, spinRate, 0.02);
    const signs = curve.map((M) => Math.sign(M[IMID]));
    expect(signs.some((s) => s !== signs[0])).toBe(true);
  });

  it("for a stable (imin/imax) start, stays a bounded small loop -- never approaches the opposite pole", () => {
    for (const axis of [IMIN, IMAX]) {
      // Compared against THIS axis's own baseline (I[axis]*spinRate), not
      // R_cas -- R_cas is anchored to imid specifically and can differ from
      // imin/imax's own natural scale by an order of magnitude (see kickedIC).
      const baseline = I[axis] * spinRate;
      const curve = currentAxisOrbitCurve(I, R_cas, IMIN, IMID, IMAX, axis, spinRate, 0.02);
      for (const M of curve) {
        expect(Math.abs(M[axis])).toBeGreaterThan(baseline * 0.5);
      }
    }
  });
});

describe("buildBackgroundWithCurrentOrbit", () => {
  const R_cas = casimirRadius(I, IMID, 1.5);
  const spinRate = 2 * Math.PI;

  it("folds the current-orbit curve into trajectories (5 total) when kickFraction > 0", () => {
    const background = buildBackgroundWithCurrentOrbit(I, R_cas, IMIN, IMAX, IMID, IMID, spinRate, 0.02);
    expect(background.trajectories).toHaveLength(5);
  });

  it("adds nothing extra when kickFraction is 0 (still just the base 4)", () => {
    const background = buildBackgroundWithCurrentOrbit(I, R_cas, IMIN, IMAX, IMID, IMID, spinRate, 0);
    expect(background.trajectories).toHaveLength(4);
  });
});
