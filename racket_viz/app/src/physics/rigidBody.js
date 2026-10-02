/**
 * Rigid-body dynamics core: quaternion utilities + Euler's equations.
 *
 * Ported from physics/rigid_body.py. Runs entirely independently of
 * Three.js -- RacketScene.js has its own `quaternionToMatrix` for building a
 * THREE.Matrix4 from a frame's quaternion; this module's `quatToRotationMatrix`
 * returns a plain 3x3 array, needed internally by controllers.js (R must be a
 * plain array there, not a Three.js type).
 *
 * scipy.integrate.solve_ivp's adaptive RK45 becomes a fixed-step classical
 * RK4 with substeps (matching how the old visualizer/server.py's _euler_rk4
 * already did this for real-time use) -- these ODEs are smooth, not stiff,
 * so a fine fixed step is sufficient; see rigidBody.test.js for the
 * conservation-law tolerances this needs to hit.
 *
 * Substep count is derived from MAX_STEP (a fixed cap on the actual RK4 step
 * size h), not from a fixed divisor of the output sample interval. A fixed
 * divisor (e.g. "10 substeps per output sample") silently produces a coarse,
 * inaccurate h whenever callers ask for a large T with a small N -- exactly
 * what phasePortrait.js's separatrix curves do (T=220, N=900, dtSample~0.24s)
 * -- which measured ~47% relative |M| drift by t=220s and made those curves
 * visibly spiral/wander ("look like wiring") instead of closing cleanly.
 * MAX_STEP=0.001 was chosen because it's exactly the effective step size the
 * main animated trajectory already uses by default (T=300s, N=30000,
 * dtSample~0.01s, 10 substeps) and which was already visually/numerically
 * verified good -- so this is a no-op for that path and only densifies the
 * under-resolved long-T/small-N background-curve case.
 *
 * Quaternion convention: q = [q0, q1, q2, q3], scalar-first.
 */
const MAX_STEP = 0.001;

/** @param {number} dtSample @returns {number} substeps giving h <= MAX_STEP */
function defaultSubsteps(dtSample) {
  return Math.max(1, Math.ceil(dtSample / MAX_STEP));
}

/** @param {number[]} q @returns {number[]} q scaled to unit norm */
export function normalizeQuaternion(q) {
  const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** @param {number[]} q scalar-first quaternion @returns {number[][]} 3x3 rotation matrix */
export function quatToRotationMatrix(q) {
  const [q0, q1, q2, q3] = normalizeQuaternion(q);
  return [
    [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
    [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
    [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
  ];
}

/** @param {number[]} q @param {number[]} w @returns {number[]} q_dot */
export function quaternionDerivative(q, w) {
  const [q0, q1, q2, q3] = q;
  const [w1, w2, w3] = w;
  return [
    0.5 * (-q1 * w1 - q2 * w2 - q3 * w3),
    0.5 * (q0 * w1 + q2 * w3 - q3 * w2),
    0.5 * (q0 * w2 + q3 * w1 - q1 * w3),
    0.5 * (q0 * w3 + q1 * w2 - q2 * w1),
  ];
}

/** state = [w1,w2,w3,q0,q1,q2,q3]; Euler's equations, vector form:
 * I*wdot = tau - w x (I*w). */
function stateDerivative(state, I, torqueFn, t) {
  const w = [state[0], state[1], state[2]];
  const q = [state[3], state[4], state[5], state[6]];
  const tau = torqueFn ? torqueFn(t, w, q) : [0, 0, 0];

  const Iw = [I[0] * w[0], I[1] * w[1], I[2] * w[2]];
  const wCrossIw = [
    w[1] * Iw[2] - w[2] * Iw[1],
    w[2] * Iw[0] - w[0] * Iw[2],
    w[0] * Iw[1] - w[1] * Iw[0],
  ];
  const wdot = [
    (tau[0] - wCrossIw[0]) / I[0],
    (tau[1] - wCrossIw[1]) / I[1],
    (tau[2] - wCrossIw[2]) / I[2],
  ];
  const qdot = quaternionDerivative(q, w);
  return [wdot[0], wdot[1], wdot[2], qdot[0], qdot[1], qdot[2], qdot[3]];
}

function rk4Step(state, I, torqueFn, t, h) {
  const k1 = stateDerivative(state, I, torqueFn, t);
  const s2 = state.map((s, i) => s + 0.5 * h * k1[i]);
  const k2 = stateDerivative(s2, I, torqueFn, t + 0.5 * h);
  const s3 = state.map((s, i) => s + 0.5 * h * k2[i]);
  const k3 = stateDerivative(s3, I, torqueFn, t + 0.5 * h);
  const s4 = state.map((s, i) => s + h * k3[i]);
  const k4 = stateDerivative(s4, I, torqueFn, t + h);
  return state.map((s, i) => s + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

/**
 * @param {number[]} w0 initial angular velocity
 * @param {number[]} I principal moments of inertia [I1,I2,I3]
 * @param {number} T total integration time
 * @param {number} N number of output samples (evenly spaced over [0,T])
 * @param {object} [opts]
 * @param {(t:number, w:number[], q:number[]) => number[]} [opts.torqueFn] body-frame
 *   external torque; omitted/undefined recovers Euler's torque-free equations
 * @param {number[]} [opts.q0] initial orientation quaternion, defaults to identity
 * @param {number} [opts.substepsPerSample] RK4 substeps between each output
 *   sample; defaults to enough substeps to keep h <= MAX_STEP regardless of
 *   dtSample (accuracy/perf knob; no scipy-style error control here)
 * @returns {{ omegas: number[][], quats: number[][] }}
 */
export function integrateFull(w0, I, T, N, opts = {}) {
  const { torqueFn = null, q0 = [1, 0, 0, 0] } = opts;

  let state = [w0[0], w0[1], w0[2], q0[0], q0[1], q0[2], q0[3]];
  const qn0 = normalizeQuaternion(q0);
  const omegas = [[state[0], state[1], state[2]]];
  const quats = [qn0];

  const dtSample = N > 1 ? T / (N - 1) : 0;
  const substepsPerSample = opts.substepsPerSample ?? defaultSubsteps(dtSample);
  const h = dtSample / substepsPerSample;
  let t = 0;

  for (let i = 1; i < N; i++) {
    for (let s = 0; s < substepsPerSample; s++) {
      state = rk4Step(state, I, torqueFn, t, h);
      t += h;
    }
    const qn = normalizeQuaternion([state[3], state[4], state[5], state[6]]);
    state[3] = qn[0];
    state[4] = qn[1];
    state[5] = qn[2];
    state[6] = qn[3];
    omegas.push([state[0], state[1], state[2]]);
    quats.push(qn);
  }

  return { omegas, quats };
}

/**
 * Torque-free angular-momentum-only integration (for static phase-portrait
 * background curves -- no orientation needed).
 * @param {number[]} M0
 * @param {number[]} I
 * @param {number} T
 * @param {number} N
 * @param {object} [opts]
 * @param {number} [opts.substepsPerSample] see integrateFull -- defaults to
 *   enough substeps to keep h <= MAX_STEP regardless of dtSample
 * @returns {number[][]}
 */
export function integrateM(M0, I, T, N, opts = {}) {
  const dMdt = (M) => [
    (1 / I[2] - 1 / I[1]) * M[1] * M[2],
    (1 / I[0] - 1 / I[2]) * M[2] * M[0],
    (1 / I[1] - 1 / I[0]) * M[0] * M[1],
  ];

  const dtSample = N > 1 ? T / (N - 1) : 0;
  const substeps = opts.substepsPerSample ?? defaultSubsteps(dtSample);
  const h = dtSample / substeps;

  let M = [M0[0], M0[1], M0[2]];
  const out = [[...M]];
  for (let i = 1; i < N; i++) {
    for (let s = 0; s < substeps; s++) {
      const k1 = dMdt(M);
      const M2 = M.map((m, idx) => m + 0.5 * h * k1[idx]);
      const k2 = dMdt(M2);
      const M3 = M.map((m, idx) => m + 0.5 * h * k2[idx]);
      const k3 = dMdt(M3);
      const M4 = M.map((m, idx) => m + h * k3[idx]);
      const k4 = dMdt(M4);
      M = M.map((m, idx) => m + (h / 6) * (k1[idx] + 2 * k2[idx] + 2 * k3[idx] + k4[idx]));
    }
    out.push([...M]);
  }
  return out;
}
