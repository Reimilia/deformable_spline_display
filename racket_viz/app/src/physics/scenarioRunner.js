/**
 * Builds one scenario -- a single continuous trajectory -- from sidebar
 * parameters, entirely in JS (no server, no precomputed file). The returned
 * object matches the exact schema DataLoader.validateScenario already
 * enforces (see ARCHITECTURE.md), which is what lets the existing render
 * pipeline (RacketScene/SphereScene/TimeSeriesPanel/ControlPanel, all built
 * against fetched static JSON) accept this with zero changes.
 */
import { buildRacket } from "./racketGeometry.js";
import { casimirRadius, energyAtAxes, buildBackgroundWithCurrentOrbit, kickedIC } from "./phasePortrait.js";
import { integrateFull, quatToRotationMatrix } from "./rigidBody.js";
import { PDBodyFrameController, defaultKpForAxisControl } from "./controllers.js";
import { sampleDisturbanceTorque } from "./wind.js";

// Shared with main.js's live sphere-background preview, so both agree on
// exactly the same reference spin rate (there's no sidebar control for it).
export const DEFAULT_SPIN_RATE = 2 * Math.PI;

/**
 * @param {object} params
 * @param {number} [params.handleLength]
 * @param {number} [params.hoopLength]
 * @param {number} [params.hoopWidth]
 * @param {number} [params.totalMass]
 * @param {number} [params.hoopFraction]
 * @param {"imin"|"imid"|"imax"} params.startAxis
 * @param {number} [params.startingPerturbation] fraction of the start axis's
 *   own baseline angular momentum (I[axis]*spinRate) to kick off-axis by; 0
 *   gives an exact on-axis start
 * @param {boolean} params.controlOn
 * @param {"imin"|"imid"|"imax"} [params.endAxis] required if controlOn
 * @param {boolean} params.windOn
 * @param {number} [params.windStd] required if windOn
 * @param {number} [params.spinRate]
 * @param {number} [params.T]
 * @param {number} [params.N]
 * @param {() => number} [params.rng] injectable uniform RNG for deterministic tests
 * @param {string} [params.label]
 * @param {string} [params.caption]
 * @returns {{ meta: object, frames: object, background: object, geometry: object }}
 */
export function buildScenario(params) {
  const {
    handleLength,
    hoopLength,
    hoopWidth,
    totalMass,
    hoopFraction,
    startAxis,
    startingPerturbation = 0.02,
    controlOn,
    endAxis,
    windOn,
    windStd,
    spinRate = DEFAULT_SPIN_RATE,
    T = 6.0,
    N = 600,
    rng = Math.random,
    label = controlOn ? "Control on" : "Free spin",
    caption = "",
    dotColor,
  } = params;

  const geometry = buildRacket({ handleLength, hoopLength, hoopWidth, totalMass, hoopFraction });
  const { I, imin, imid, imax } = geometry;
  const axisIndex = { imin, imid, imax };

  // R_cas is purely a DISPLAY scale (it never feeds into the actual
  // dynamics -- controller gains come from I/spinRate directly, and
  // desired_H/desired_L below are computed from omegaStar/I, not R_cas), so
  // this only affects where things land in the sphere panel, never whether
  // control actually works.
  //
  // Free (uncontrolled) runs conserve |M| exactly at whatever |M0| actually
  // is -- I[startAxis]*spinRate (see kickedIC), not necessarily
  // I[imid]*spinRate now that a perturbation is available from any start
  // axis. Using imid unconditionally (the old behavior) drew the
  // sphere/fixed-points/trajectory family at the WRONG radius whenever
  // starting on imin/imax -- the dot sat deep inside the sphere instead of
  // near its own fixed point.
  //
  // Controlled runs aren't torque-free, so there's no single radius |M(t)|
  // stays on throughout -- but the CONTROLLER'S ACTUAL TARGET is
  // I[endAxis]*spinRate (that's what desired_L already is), so basing R_cas
  // on endAxis means a successful run's dot lands exactly on the target's
  // green/red marker instead of floating at some unrelated imid-based
  // radius. It also makes a wind-driven steady-state offset finally look
  // like what it is -- a small, visible miss near the target marker --
  // instead of being swamped by an unrelated scale mismatch.
  const R_cas = casimirRadius(I, axisIndex[controlOn ? endAxis : startAxis], spinRate);

  // Always applied (previously only for imid, and only when control/wind
  // were both off, since starting exactly on the unstable axis with nothing
  // to perturb it would just sit at a boring fixed point forever). Now
  // user-controlled and available from any axis, so you can also see a
  // near-stable-axis orbit, not just the imid flip -- wind (below) still
  // layers on top independently, unaffected by this.
  const M0 = kickedIC(I, imin, imid, imax, axisIndex[startAxis], spinRate, startingPerturbation);
  const w0 = [M0[0] / I[0], M0[1] / I[1], M0[2] / I[2]];

  const windVec = windOn ? sampleDisturbanceTorque(windStd, rng) : [0, 0, 0];

  let controller = null;
  let omegaStar = null;
  if (controlOn) {
    omegaStar = [0, 0, 0];
    omegaStar[axisIndex[endAxis]] = spinRate;
    const Kp = defaultKpForAxisControl(I, spinRate);
    controller = new PDBodyFrameController(omegaStar, Kp);
  }

  const torqueFn =
    controlOn || windOn
      ? (t, w, q) => {
          const R = quatToRotationMatrix(q);
          const ctrlTau = controller ? controller.call(R, w) : [0, 0, 0];
          return [ctrlTau[0] + windVec[0], ctrlTau[1] + windVec[1], ctrlTau[2] + windVec[2]];
        }
      : null;

  const { omegas, quats } = integrateFull(w0, I, T, N, { torqueFn });
  const t = Array.from({ length: N }, (_, i) => (N > 1 ? (T * i) / (N - 1) : 0));
  const M_body = omegas.map((w) => [w[0] * I[0], w[1] * I[1], w[2] * I[2]]);

  // Recorded post-hoc (one extra call per output sample, not per RK4 substep)
  // rather than captured during integration -- the controller's own output at
  // each sample's actual (R, omega) is exactly "the control function" the user
  // wants displayed, and doesn't depend on which of RK4's 4 intermediate
  // evaluations you'd otherwise have to pick. Deliberately controller-only
  // (excludes wind), matching "control function" -- wind is a separate,
  // constant disturbance, not part of what the controller is doing.
  const controllerTorque = controlOn
    ? omegas.map((w, i) => controller.call(quatToRotationMatrix(quats[i]), w))
    : omegas.map(() => [0, 0, 0]);

  const background = buildBackgroundWithCurrentOrbit(
    I,
    R_cas,
    imin,
    imax,
    imid,
    axisIndex[startAxis],
    spinRate,
    startingPerturbation
  );

  const meta = {
    mode: controlOn ? "controlled" : "free",
    I,
    R_cas,
    segments: [
      {
        start: 0,
        end: N,
        label,
        caption,
        dot_color: dotColor ?? (controlOn ? "#ffaa00" : "#44ee66"),
      },
    ],
  };
  // wind_on lets rendering code determine whether "achieved target" is even a
  // well-posed question for this run (see ControlMetrics.js): holding an
  // unstable axis, or holding anything against a constant disturbance, never
  // settles to zero corrective torque, so there's no single "done" instant --
  // only ever-ongoing correction.
  meta.wind_on = windOn;
  if (controlOn) {
    meta.target_axis = axisIndex[endAxis];
    meta.desired_H = 0.5 * (omegaStar[0] ** 2 * I[0] + omegaStar[1] ** 2 * I[1] + omegaStar[2] ** 2 * I[2]);
    meta.desired_L = Math.sqrt(
      (omegaStar[0] * I[0]) ** 2 + (omegaStar[1] * I[1]) ** 2 + (omegaStar[2] * I[2]) ** 2
    );
  } else {
    meta.H_axis = energyAtAxes(I, R_cas);
  }

  return {
    meta,
    frames: { t, quaternion: quats, M_body, controller_torque: controllerTorque },
    background: {
      trajectories: background.trajectories,
      separatrices: background.separatrices,
      stable_fixed_points: background.stableFixedPoints,
      unstable_fixed_points: background.unstableFixedPoints,
    },
    geometry: {
      verts_body: geometry.vertsBody,
      handle_idx: geometry.handleIdx,
      hoop_idx: geometry.hoopIdx,
      face_thickness: geometry.faceThickness,
      face_normal_body: geometry.faceNormalBody,
    },
  };
}
