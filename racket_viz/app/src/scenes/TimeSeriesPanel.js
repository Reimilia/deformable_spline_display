/**
 * Panel 3: angular velocity vs. time -- matches summer-2026/rotate.py's
 * `ax_omega` panel (three colored ω1/ω2/ω3 lines). Plain HTML5 canvas 2D,
 * not Three.js: a 2D line chart doesn't need a 3D scene, and adding a charting
 * library wasn't part of the approved architecture.
 *
 * Color convention (matches the sphere panel's existing stable/unstable
 * fixed-point colors -- see theme.js): omega[imin] THEME.stable, omega[imid]
 * THEME.unstable (the unstable axis), omega[imax] THEME.imax.
 */
import { computeVisibleWindow } from "./rollingWindow.js";
import { drawAxes, computeNiceStep } from "./axisTicks.js";
import { THEME } from "../theme.js";

export const OMEGA_COLORS = [THEME.stable, THEME.unstable, THEME.imax];

/**
 * @param {[number, number, number][]} M_body per-frame angular momentum
 * @param {[number, number, number]} I principal moments of inertia
 * @returns {[number, number, number][]} per-frame angular velocity
 */
export function computeOmega(M_body, I) {
  const [I1, I2, I3] = I;
  return M_body.map(([m1, m2, m3]) => [m1 / I1, m2 / I2, m3 / I3]);
}

/**
 * @param {[number, number, number][]} omegaSeries
 * @param {number} padFactor fraction of the range to pad above/below
 * @returns {[number, number]} [min, max] y-axis domain
 */
export function computeOmegaDomain(omegaSeries, padFactor = 0.1) {
  let min = Infinity;
  let max = -Infinity;
  for (const [a, b, c] of omegaSeries) {
    min = Math.min(min, a, b, c);
    max = Math.max(max, a, b, c);
  }
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * padFactor;
  return [min - pad, max + pad];
}

/**
 * Draws the full panel 3 chart: three omega lines over a scrolling window of
 * time (see rollingWindow.js -- a long run's entire history no longer gets
 * squeezed into one fixed panel width), and a colored marker per line at the
 * current value (always at the window's right edge).
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} opts
 * @param {number[]} opts.t
 * @param {[number, number, number][]} opts.omega
 * @param {[number, number]} opts.domain
 * @param {number} opts.currentIndex
 * @param {number} opts.width
 * @param {number} opts.height
 * @param {number} [opts.windowSeconds=5]
 */
export function drawTimeSeries(ctx, { t, omega, domain, currentIndex, width, height, windowSeconds = 5 }) {
  const [yMin, yMax] = domain;
  const { startIdx, endIdx, windowStart, windowEnd } = computeVisibleWindow(t, currentIndex, windowSeconds);

  const marginLeft = 46;
  const marginRight = 18;
  const marginTop = 10;
  const marginBottom = 34;
  const plotW = width - marginLeft - marginRight;
  const plotH = height - marginBottom - marginTop;
  const plotRight = width - marginRight;
  const plotBottom = marginTop + plotH;
  const toX = (ti) => marginLeft + ((ti - windowStart) / (windowEnd - windowStart || 1)) * plotW;
  const toY = (v) => marginTop + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = THEME.panelBg;
  ctx.fillRect(0, 0, width, height);

  const { majorStep, minorStep } = computeNiceStep(yMax - yMin);
  drawAxes(ctx, {
    toX,
    toY,
    xMin: windowStart,
    xMax: windowEnd,
    yMin,
    yMax,
    yMinorStep: minorStep,
    yMajorStep: majorStep,
    plotLeft: marginLeft,
    plotRight,
    plotTop: marginTop,
    plotBottom,
    xLabel: "t (s)",
    yLabel: "ω (rad/s)",
  });

  // zero line
  if (yMin < 0 && yMax > 0) {
    ctx.strokeStyle = THEME.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(marginLeft, toY(0));
    ctx.lineTo(plotRight, toY(0));
    ctx.stroke();
  }

  for (let comp = 0; comp < 3; comp++) {
    ctx.strokeStyle = OMEGA_COLORS[comp];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = startIdx; i <= endIdx; i++) {
      const x = toX(t[i]);
      const y = toY(omega[i][comp]);
      if (i === startIdx) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Current-value marker per line, at "now"'s actual x position -- NOT
  // hardcoded to the right edge: during ramp-up (run younger than the
  // window) `toX(t[currentIndex])` is still less than `width` and advances
  // left-to-right, only reaching the edge exactly when the window starts
  // sliding (see rollingWindow.js).
  const nowX = toX(t[currentIndex]);
  for (let comp = 0; comp < 3; comp++) {
    const cy = toY(omega[currentIndex][comp]);
    ctx.fillStyle = THEME.current;
    ctx.beginPath();
    ctx.arc(nowX, cy, 3.5, 0, 2 * Math.PI);
    ctx.fill();
  }
}
