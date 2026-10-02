import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { normalizeQuaternion, quatToRotationMatrix, integrateFull, integrateM } from "../rigidBody.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const I_DEFAULT = [0.075, 0.875, 0.95];

function norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

describe("normalizeQuaternion", () => {
  it("always returns unit norm", () => {
    for (const q of [[1, 0, 0, 0], [2, 0, 0, 0], [1, 1, 1, 1], [0.3, -0.1, 5.0, 2.0]]) {
      expect(norm(normalizeQuaternion(q))).toBeCloseTo(1.0, 9);
    }
  });
});

describe("quatToRotationMatrix matches the shared Python quat_to_R fixture", () => {
  const fixturePath = path.join(
    __dirname,
    "../../../../physics/tests/fixtures/quaternion_rotation_pairs.json"
  );
  const cases = JSON.parse(readFileSync(fixturePath, "utf-8"));

  for (const { name, q, R } of cases) {
    it(name, () => {
      const got = quatToRotationMatrix(q);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(got[i][j]).toBeCloseTo(R[i][j], 9);
        }
      }
    });
  }
});

describe("integrateFull -- torque-free", () => {
  it("conserves energy H = 0.5 * w.(I*w)", () => {
    const trials = [
      [0.5, 1.2, 0.3],
      [1.0, 0.1, 2.0],
    ];
    for (const w0 of trials) {
      const { omegas } = integrateFull(w0, I_DEFAULT, 10.0, 200);
      const H = omegas.map((w) => 0.5 * (w[0] ** 2 * I_DEFAULT[0] + w[1] ** 2 * I_DEFAULT[1] + w[2] ** 2 * I_DEFAULT[2]));
      const H0 = H[0];
      for (const h of H) {
        expect(h / H0).toBeCloseTo(1, 5);
      }
    }
  });

  it("conserves |M| = |I*w| (the Casimir invariant)", () => {
    const w0 = [0.8, 1.5, 0.2];
    const { omegas } = integrateFull(w0, I_DEFAULT, 10.0, 200);
    const Mmags = omegas.map((w) => norm([w[0] * I_DEFAULT[0], w[1] * I_DEFAULT[1], w[2] * I_DEFAULT[2]]));
    const M0 = Mmags[0];
    for (const m of Mmags) {
      expect(m / M0).toBeCloseTo(1, 5);
    }
  });

  it("fixed points (pure spin on a stable axis) are stationary", () => {
    const R_cas = 1.5;
    for (const axis of [0, 2]) {
      const w0 = [0, 0, 0];
      w0[axis] = R_cas / I_DEFAULT[axis];
      const { omegas } = integrateFull(w0, I_DEFAULT, 5.0, 100);
      for (const w of omegas) {
        for (let i = 0; i < 3; i++) {
          expect(w[i]).toBeCloseTo(w0[i], 6);
        }
      }
    }
  });

  it("a Dzhanibekov flip occurs near the unstable (imid) axis", () => {
    // Equal-in-omega perturbation, NOT equal-in-M -- the saddle at the
    // unstable fixed point has a specific growing-eigendirection ratio
    // (~11.6:1 between the two stable-axis perturbations for this I), and an
    // equal-in-M kick landed almost exactly on the *decaying* eigendirection
    // during the Python physics work (see DECISIONS.md), taking impractically
    // long to show a flip. Equal-in-omega generically avoids that.
    const R_cas = 1.5;
    const spin = R_cas / I_DEFAULT[1];
    const eps = 0.02;
    const w0 = [eps, spin, eps];
    const { omegas } = integrateFull(w0, I_DEFAULT, 15.0, 1500);
    const signs = omegas.map((w) => Math.sign(w[0]));
    expect(signs.some((s) => s !== signs[0])).toBe(true);
  });
});

describe("integrateM -- long-run drift regression", () => {
  it("conserves |M| within 1% over the actual production separatrix config (T=220, N=900)", () => {
    // Regression test for a real bug: a hardcoded substeps=10 gave h~0.0245s
    // here (dtSample=220/899 substepped only 10x), measured at ~47% relative
    // |M| drift by t=220s -- exactly why the separatrix curves looked like
    // "wiring" instead of closing cleanly. Substep count is now derived from
    // a fixed max step size instead of a fixed divisor of dtSample.
    const I = [0.075, 0.875, 0.95];
    const R_cas = 1.5;
    const eps = 0.001 * R_cas;
    const M0 = [eps, Math.sqrt(R_cas ** 2 - 2 * eps ** 2), eps]; // near-separatrix IC (imid dominant)
    const out = integrateM(M0, I, 220.0, 900);
    const M0mag = Math.hypot(...out[0]);
    for (const M of out) {
      const mag = Math.hypot(...M);
      expect(Math.abs(mag / M0mag - 1)).toBeLessThan(0.01);
    }
  });

  it("respects an explicit substepsPerSample override", () => {
    const I = [0.075, 0.875, 0.95];
    const out = integrateM([0.1, 1.0, 0.1], I, 1.0, 10, { substepsPerSample: 1 });
    expect(out.length).toBe(10);
  });
});
