import { describe, expect, it } from "vitest";

import {
  hat,
  vee,
  expmSO3,
  logmSO3,
  PDBodyFrameController,
  GeometricAttitudeController,
  defaultKpForAxisControl,
} from "../controllers.js";
import { integrateFull, quatToRotationMatrix, normalizeQuaternion } from "../rigidBody.js";

const I_DEFAULT = [0.075, 0.875, 0.95];
// Calibrated numerically against I_DEFAULT in the Python physics work (see
// DECISIONS.md) -- NOT the LieSPHGP PR's own cfg0 numbers, which used a very
// different inertia scale (I ~ 0.006-0.013 there vs. I_DEFAULT's 0.075-0.95).
const STAGE_D_KP = 1.0;
const STAGE_D_T = 6.0;

function norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

describe("hat/vee", () => {
  it("are inverses", () => {
    const vs = [
      [1, 2, 3],
      [-0.5, 0.2, 4],
    ];
    for (const v of vs) {
      const back = vee(hat(v));
      for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(v[i], 10);
    }
  });
});

describe("expmSO3/logmSO3", () => {
  it("are inverses away from the antipodal set", () => {
    const vs = [
      [0.3, 0, 0],
      [0, 1.2, 0],
      [0.2, -0.4, 0.6],
    ];
    for (const v of vs) {
      const R = expmSO3(hat(v));
      const back = vee(logmSO3(R));
      for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(v[i], 6);
    }
  });
});

describe("PDBodyFrameController (Stage D)", () => {
  it("converges an axis-2 spin-up within the calibrated time/gain", () => {
    const omegaStar = [0, 0, 2 * Math.PI];
    const ctrl = new PDBodyFrameController(omegaStar, STAGE_D_KP);
    const torqueFn = (t, w, q) => ctrl.call(quatToRotationMatrix(q), w);

    const w0 = [0, 2 * Math.PI, 0];
    const { omegas } = integrateFull(w0, I_DEFAULT, STAGE_D_T, Math.round(STAGE_D_T * 100), { torqueFn });

    const final = omegas[omegas.length - 1];
    const err = norm([final[0] - omegaStar[0], final[1] - omegaStar[1], final[2] - omegaStar[2]]);
    expect(err).toBeLessThan(0.05 * norm(omegaStar));
  });

  it("H(t) and |M(t)| both change once torque is applied (not conserved)", () => {
    const omegaStar = [0, 0, 2 * Math.PI];
    const ctrl = new PDBodyFrameController(omegaStar, STAGE_D_KP);
    const torqueFn = (t, w, q) => ctrl.call(quatToRotationMatrix(q), w);

    const w0 = [0, 2 * Math.PI, 0];
    const { omegas } = integrateFull(w0, I_DEFAULT, STAGE_D_T, Math.round(STAGE_D_T * 100), { torqueFn });

    const H = (w) => 0.5 * (w[0] ** 2 * I_DEFAULT[0] + w[1] ** 2 * I_DEFAULT[1] + w[2] ** 2 * I_DEFAULT[2]);
    const Hstart = H(omegas[0]);
    const Hend = H(omegas[omegas.length - 1]);
    expect(Math.abs(Hstart - Hend)).toBeGreaterThan(1e-3);

    const Mmag = (w) => norm([w[0] * I_DEFAULT[0], w[1] * I_DEFAULT[1], w[2] * I_DEFAULT[2]]);
    expect(Math.abs(Mmag(omegas[0]) - Mmag(omegas[omegas.length - 1]))).toBeGreaterThan(1e-3);
  });
});

describe("GeometricAttitudeController (Stage F)", () => {
  it("converges attitude and rate to the target", () => {
    const Rstar = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const omegaStar = [0, 0, 0];
    const ctrl = new GeometricAttitudeController(Rstar, omegaStar, 1.0, 1.0);
    const torqueFn = (t, w, q) => ctrl.call(quatToRotationMatrix(q), w);

    const q0 = normalizeQuaternion([0.9, 0.1, 0.2, 0.1]);
    const w0 = [0.5, -0.3, 0.2];
    const T = 8.0;
    const { omegas, quats } = integrateFull(w0, I_DEFAULT, T, Math.round(T * 100), { torqueFn, q0 });

    const Rfinal = quatToRotationMatrix(quats[quats.length - 1]);
    // attitude error = vee(logm(Rstar^T @ Rfinal))
    const RtR = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 3; k++) RtR[i][j] += Rstar[k][i] * Rfinal[k][j];
      }
    }
    const attErr = norm(vee(logmSO3(RtR)));
    expect(attErr).toBeLessThan(0.05);
    expect(norm(omegas[omegas.length - 1])).toBeLessThan(0.05);
  });
});

describe("defaultKpForAxisControl", () => {
  it("scales with inertia rather than returning a fixed constant", () => {
    const smallI = [0.006, 0.011, 0.013]; // roughly the LieSPHGP cfg0 scale
    const largeI = [0.075, 0.875, 0.95]; // this project's default scale
    const kpSmall = defaultKpForAxisControl(smallI, 2 * Math.PI);
    const kpLarge = defaultKpForAxisControl(largeI, 2 * Math.PI);
    expect(kpLarge).toBeGreaterThan(kpSmall);
  });

  it("produces a gain that actually converges for a non-default geometry", () => {
    const I2 = [0.09, 1.0, 1.1];
    const omegaStar = [0, 0, 2 * Math.PI];
    const Kp = defaultKpForAxisControl(I2, norm(omegaStar));
    const ctrl = new PDBodyFrameController(omegaStar, Kp);
    const torqueFn = (t, w, q) => ctrl.call(quatToRotationMatrix(q), w);

    const w0 = [0, 2 * Math.PI, 0];
    const T = 10.0;
    const { omegas } = integrateFull(w0, I2, T, Math.round(T * 100), { torqueFn });
    const final = omegas[omegas.length - 1];
    const err = norm([final[0] - omegaStar[0], final[1] - omegaStar[1], final[2] - omegaStar[2]]);
    expect(err).toBeLessThan(0.1 * norm(omegaStar));
  });
});
