/**
 * Derived "information back to the user" readouts for controlled scenarios:
 * how much work the controller has done so far, how hard it's currently
 * pushing, and whether/when it has actually reached its target.
 */
import { computeOmega } from "./TimeSeriesPanel.js";
import { computeVisibleWindow } from "./rollingWindow.js";
import { THEME } from "../theme.js";

const IMID = 1; // this codebase's fixed convention: imin=0, imid=1, imax=2 always

/**
 * Whether "achieved target" is a well-posed, ever-settling question for this
 * scenario. Requires an active controller, no wind (a constant disturbance
 * means corrective torque never relaxes to zero), and a STABLE target axis
 * (holding the unstable imid axis against a perturbation also never lets
 * torque settle to zero -- see DECISIONS.md). When false, callers should show
 * an ongoing effort readout only, never an "achieved"/"done" flag.
 * @param {object} meta a scenario's `meta` block
 * @returns {boolean}
 */
export function isAchievable(meta) {
  return meta.mode === "controlled" && !meta.wind_on && meta.target_axis !== IMID;
}

/**
 * Cumulative work done BY THE CONTROLLER specifically (excludes wind's own
 * contribution, even when wind is also on): running (signed) integral of
 * controller_torque . omega, trapezoidal rule (rectangle rule's O(dt) error
 * was large enough during the fast initial transient to visibly miss the
 * H(t) - H(0) cross-check below). When wind is off this closely matches
 * H(t) - H(0) (a useful cross-check: dH/dt = torque . omega exactly).
 * @param {[number,number,number][]} controllerTorque
 * @param {[number,number,number][]} M_body
 * @param {number[]} I
 * @param {number[]} t
 * @returns {number[]} cumulative work at each frame; work[0] = 0
 */
export function computeCumulativeWork(controllerTorque, M_body, I, t) {
  const omega = computeOmega(M_body, I);
  const power = controllerTorque.map(
    (tau, i) => tau[0] * omega[i][0] + tau[1] * omega[i][1] + tau[2] * omega[i][2]
  );
  const work = new Array(t.length);
  work[0] = 0;
  for (let i = 1; i < t.length; i++) {
    const dt = t[i] - t[i - 1];
    work[i] = work[i - 1] + 0.5 * (power[i - 1] + power[i]) * dt;
  }
  return work;
}

/** @param {[number,number,number][]} controllerTorque @returns {number[]} */
export function computeTorqueMagnitude(controllerTorque) {
  return controllerTorque.map(([x, y, z]) => Math.hypot(x, y, z));
}

/**
 * Finds the first index at which the racket has been within `tolerance`
 * (relative to desired_L) of the controller's target angular momentum for at
 * least `sustainSeconds` continuously -- not just a lucky momentary pass
 * through the target.
 * @param {[number,number,number][]} M_body
 * @param {number[]} t
 * @param {object} meta a scenario's `meta` block
 * @param {number} [tolerance=0.05] relative to desired_L
 * @param {number} [sustainSeconds=1.0]
 * @returns {number|null} index where sustained convergence began, or null if
 *   not achievable at all, or achievable but never sustained within the run
 */
export function computeAchievedIndex(M_body, t, meta, tolerance = 0.05, sustainSeconds = 1.0) {
  if (!isAchievable(meta)) return null;

  const M_star = [0, 0, 0];
  M_star[meta.target_axis] = meta.desired_L; // pure-axis target: desired_L fully determines the vector

  let sustainedSinceIdx = null;
  for (let i = 0; i < M_body.length; i++) {
    const M = M_body[i];
    const err =
      Math.hypot(M[0] - M_star[0], M[1] - M_star[1], M[2] - M_star[2]) / meta.desired_L;
    if (err < tolerance) {
      if (sustainedSinceIdx === null) sustainedSinceIdx = i;
      if (t[i] - t[sustainedSinceIdx] >= sustainSeconds) {
        return sustainedSinceIdx;
      }
    } else {
      sustainedSinceIdx = null;
    }
  }
  return null;
}

/**
 * Draws the sidebar's small "control function" panel: torque magnitude vs.
 * time, same 10s scrolling window as panels 3 & 4, with a yellow marker at
 * the current value -- only meaningful while a controller is actually
 * running, so callers should gate showing this panel on `controlOn`.
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} opts
 * @param {number[]} opts.t
 * @param {number[]} opts.torqueMagnitude
 * @param {number} opts.currentIndex
 * @param {number} opts.width
 * @param {number} opts.height
 * @param {number} [opts.windowSeconds=5]
 */
export function drawTorquePanel(ctx, { t, torqueMagnitude, currentIndex, width, height, windowSeconds = 5 }) {
  const maxMag = Math.max(...torqueMagnitude, 1e-9);
  const yMin = 0;
  const yMax = maxMag * 1.1;

  const { startIdx, endIdx, windowStart, windowEnd } = computeVisibleWindow(t, currentIndex, windowSeconds);
  const toX = (ti) => ((ti - windowStart) / (windowEnd - windowStart || 1)) * width;
  const toY = (v) => height - ((v - yMin) / (yMax - yMin || 1)) * height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = THEME.panelBg;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = THEME.target;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = startIdx; i <= endIdx; i++) {
    const x = toX(t[i]);
    const y = toY(torqueMagnitude[i]);
    if (i === startIdx) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // "Now" marker at its actual x position, not hardcoded to the right edge --
  // see rollingWindow.js: it advances left-to-right during ramp-up and only
  // reaches the edge once the window starts sliding.
  ctx.fillStyle = THEME.current;
  ctx.beginPath();
  ctx.arc(toX(t[currentIndex]), toY(torqueMagnitude[currentIndex]), 3.5, 0, 2 * Math.PI);
  ctx.fill();
}
