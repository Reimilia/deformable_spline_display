/**
 * IDA-PBC controllers for driving the racket to a target spin state.
 * Ported from physics/controllers.py (itself adapted from LieSPHGP PR #1's
 * controller_stageD.py / controller_stageF.py).
 */

/** @param {number[]} v @returns {number[][]} skew-symmetric matrix, hat(v)@w == cross(v,w) */
export function hat(v) {
  return [
    [0, -v[2], v[1]],
    [v[2], 0, -v[0]],
    [-v[1], v[0], 0],
  ];
}

/** @param {number[][]} Omega @returns {number[]} axial vector (inverse of hat) */
export function vee(Omega) {
  return [Omega[2][1], Omega[0][2], Omega[1][0]];
}

/** @param {number[][]} Omega so(3) @returns {number[][]} SO(3), Rodrigues formula */
export function expmSO3(Omega) {
  const v = vee(Omega);
  const theta = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
  if (theta < 1e-7) {
    return [
      [1 + Omega[0][0], Omega[0][1], Omega[0][2]],
      [Omega[1][0], 1 + Omega[1][1], Omega[1][2]],
      [Omega[2][0], Omega[2][1], 1 + Omega[2][2]],
    ];
  }
  const n = [v[0] / theta, v[1] / theta, v[2] / theta];
  const hatN = hat(n);
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const R = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      R[i][j] = cos * (i === j ? 1 : 0) + sin * hatN[i][j] + (1 - cos) * n[i] * n[j];
    }
  }
  return R;
}

/** @param {number[][]} R SO(3) @returns {number[][]} so(3), ||vee(result)|| <= pi */
export function logmSO3(R) {
  const trace = R[0][0] + R[1][1] + R[2][2];
  const cosTheta = Math.max(-1, Math.min(1, (trace - 1) / 2));
  const theta = Math.acos(cosTheta);

  if (theta < 1e-7) {
    return [
      [0, (R[0][1] - R[1][0]) / 2, (R[0][2] - R[2][0]) / 2],
      [(R[1][0] - R[0][1]) / 2, 0, (R[1][2] - R[2][1]) / 2],
      [(R[2][0] - R[0][2]) / 2, (R[2][1] - R[1][2]) / 2, 0],
    ];
  }

  if (Math.abs(theta - Math.PI) < 1e-4) {
    // R is symmetric at theta=pi, so S = (R+I)/2 = n n^T exactly.
    const S = [
      [(R[0][0] + 1) / 2, R[0][1] / 2, R[0][2] / 2],
      [R[1][0] / 2, (R[1][1] + 1) / 2, R[1][2] / 2],
      [R[2][0] / 2, R[2][1] / 2, (R[2][2] + 1) / 2],
    ];
    const diag = [S[0][0], S[1][1], S[2][2]];
    let i = 0;
    if (diag[1] > diag[i]) i = 1;
    if (diag[2] > diag[i]) i = 2;
    const denom = Math.sqrt(Math.max(S[i][i], 1e-15));
    const n = [S[0][i] / denom, S[1][i] / denom, S[2][i] / denom];
    const hatN = hat(n);
    return hatN.map((row) => row.map((x) => x * theta));
  }

  const factor = theta / (2 * Math.sin(theta));
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) out[i][j] = factor * (R[i][j] - R[j][i]);
  }
  return out;
}

function clipVec(v, clip) {
  return v.map((x) => Math.max(-clip, Math.min(clip, x)));
}

export class PDBodyFrameController {
  /** @param {number[]} omegaStar @param {number} [Kp=0.10] @param {number} [clip=2.0] */
  constructor(omegaStar, Kp = 0.1, clip = 2.0) {
    this.omegaStar = [...omegaStar];
    this.Kp = Kp;
    this.clip = clip;
  }

  /** @param {number[][]} R unused in Stage D, kept for a uniform call signature
   * @param {number[]} omega @returns {number[]} body-frame torque */
  call(R, omega) {
    const tau = [0, 1, 2].map((i) => this.Kp * (this.omegaStar[i] - omega[i]));
    return clipVec(tau, this.clip);
  }
}

export class GeometricAttitudeController {
  /** @param {number[][]} Rstar @param {number[]} omegaStar @param {number} [KR=0.10] @param {number} [Kp=0.10] @param {number} [clip=2.0] */
  constructor(Rstar, omegaStar, KR = 0.1, Kp = 0.1, clip = 2.0) {
    this.Rstar = Rstar.map((row) => [...row]);
    this.omegaStar = [...omegaStar];
    this.KR = KR;
    this.Kp = Kp;
    this.clip = clip;
  }

  /** @param {number[][]} R @param {number[]} omega @returns {number[]} body-frame torque */
  call(R, omega) {
    const RtR = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += this.Rstar[k][i] * R[k][j];
        RtR[i][j] = s;
      }
    }
    const eR = vee(logmSO3(RtR));
    const eW = [0, 1, 2].map((i) => omega[i] - this.omegaStar[i]);
    const tau = [0, 1, 2].map((i) => -this.KR * eR[i] - this.Kp * eW[i]);
    return clipVec(tau, this.clip);
  }
}

/**
 * A default proportional gain for PDBodyFrameController that scales with the
 * racket's actual inertia, rather than reusing a constant calibrated for one
 * specific geometry (see physics/tests/test_controllers.py::STAGE_D_KP and
 * DECISIONS.md -- that constant was itself a recalibration off a different
 * inertia scale, so hardcoding it again here would silently misbehave for a
 * user-chosen geometry/mass far from the default).
 *
 * Kp_min formula from the tennis-racket-effect branch's simulate_stageE.py
 * docstring (derived for the hardest case, stabilizing the unstable
 * intermediate axis): Kp_min = omega* * sqrt(|I2-I3| * |I1-I2|). Returns
 * Kp_min scaled by a safety factor for a comfortably fast, non-marginal
 * response, regardless of which axis is actually being targeted.
 *
 * @param {number[]} I [I1,I2,I3]
 * @param {number} omegaStarMag target angular speed magnitude
 * @param {number} [safetyFactor=3]
 * @returns {number}
 */
export function defaultKpForAxisControl(I, omegaStarMag, safetyFactor = 3) {
  const kpMin = omegaStarMag * Math.sqrt(Math.abs(I[1] - I[2]) * Math.abs(I[0] - I[1]));
  return safetyFactor * kpMin;
}
