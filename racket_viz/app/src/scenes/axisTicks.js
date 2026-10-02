/**
 * Shared tick-marked axis drawing for the 2D canvas panels (energy panel in
 * ControlPanel.js, angular-velocity panel in TimeSeriesPanel.js). Both panels
 * already compute an exact toX/toY linear map and x/y domain for their data;
 * this just draws grid/ticks/labels using those same values, so it doesn't
 * duplicate that math -- callers still own their own domain and toX/toY.
 *
 * The x-axis (time, a continuously sliding window -- see rollingWindow.js)
 * uses evenly-spaced ticks across whatever the current window happens to be
 * (niceTicks). The y-axis is physical (energy, angular velocity) and reads
 * better anchored at fixed, physically-meaningful round numbers rather than
 * ticks that shift with the data range -- see stepTicks: minor ticks (every
 * yMinorStep) get a tick mark only, major ticks (every yMinorStep) also get
 * a gridline and a numeric label.
 */

import { THEME } from "../theme.js";

const AXIS_LINE_COLOR = THEME.border;
const GRID_LINE_COLOR = THEME.gridLine;
const LABEL_COLOR = THEME.muted;
const TICK_LENGTH = 4;
const MINOR_TICK_LENGTH = 2;

/**
 * Evenly-spaced tick values spanning [min, max], inclusive of both ends.
 * Not a "round number" (d3-style) algorithm -- just min/max split into
 * `count` equal steps -- deliberately simple for this pass.
 * @param {number} min
 * @param {number} max
 * @param {number} [count=5]
 * @returns {number[]}
 */
export function niceTicks(min, max, count = 5) {
  if (min === max) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + i * step);
}

/**
 * Tick values at every multiple of `step` covering [min, max] -- anchored at
 * 0 (so e.g. step=2 always lands on ...,-2,0,2,4,..., never on an arbitrary
 * offset determined by min).
 * @param {number} min
 * @param {number} max
 * @param {number} step
 * @returns {number[]}
 */
export function stepTicks(min, max, step) {
  if (!step || step <= 0) return [];
  const start = Math.ceil(min / step - 1e-9) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 1e-6; v += step) {
    ticks.push(Math.round(v / step) * step || 0); // `|| 0` folds -0 to 0
  }
  return ticks;
}

/**
 * Picks a "nice" (1/2/5 x a power of ten) major step size that puts roughly
 * `targetCount` labeled ticks across `range`, plus a minor (unlabeled)
 * subdivision of that step. Anchored at 0 via stepTicks, so it always lines
 * up with a round number regardless of where the data actually starts.
 *
 * Bucket thresholds are the geometric means between neighboring "nice"
 * residuals (1, 2, 5, 10) -- the standard rounding rule (as in D3/matplotlib)
 * for picking whichever of them a given rough step is numerically closest
 * to on a log scale. `targetCount=8` (not the more typical 5) is chosen
 * specifically because it reproduces this app's two previously hand-tuned
 * defaults exactly: omega's range (~16 rad/s) -> major=2/minor=1, and the
 * default geometry's H range (~220) -> major=20/minor=10.
 *
 * This replaces a fixed hardcoded step (tuned for one specific geometry's H
 * range) that left charts with only a single "0" tick visible whenever the
 * actual range was much smaller than whatever the fixed step assumed.
 *
 * @param {number} range yMax - yMin
 * @param {number} [targetCount=8]
 * @returns {{ majorStep: number, minorStep: number }}
 */
export function computeNiceStep(range, targetCount = 8) {
  if (!(range > 0)) return { majorStep: 1, minorStep: 0.5 };
  const rough = range / targetCount;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const residual = rough / magnitude;
  let niceResidual;
  let minorDivisor;
  if (residual < Math.SQRT2) {
    niceResidual = 1;
    minorDivisor = 2;
  } else if (residual < Math.sqrt(10)) {
    niceResidual = 2;
    minorDivisor = 2;
  } else if (residual < Math.sqrt(50)) {
    niceResidual = 5;
    minorDivisor = 5;
  } else {
    niceResidual = 10;
    minorDivisor = 2;
  }
  const majorStep = niceResidual * magnitude;
  return { majorStep, minorStep: majorStep / minorDivisor };
}

/**
 * Whether a stepTicks value falls on a labeled ("major") tick -- a multiple
 * of majorStep, which is itself a multiple of the minor step used to
 * generate `value`.
 * @param {number} value
 * @param {number} majorStep
 * @returns {boolean}
 */
export function isMajorTick(value, majorStep) {
  const n = Math.round(value / majorStep);
  return Math.abs(value - n * majorStep) < majorStep * 1e-6;
}

/**
 * Formats a tick value adaptively: enough decimals to distinguish
 * neighboring ticks (based on the span between them), capped so labels
 * stay short.
 * @param {number} value
 * @param {number} span the value range the ticks are drawn over
 * @returns {string}
 */
export function formatTick(value, span) {
  if (span === 0) return value.toFixed(2);
  const decimals = Math.max(0, Math.min(3, 2 - Math.floor(Math.log10(span))));
  return value.toFixed(decimals);
}

/**
 * Formats a stepTicks value: decimals are derived from the step size itself
 * (not the domain span), so a step of 2 or 10 always prints as an integer
 * ("2", "10"), not "2.00".
 * @param {number} value
 * @param {number} step
 * @returns {string}
 */
export function formatStepTick(value, step) {
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return value.toFixed(decimals);
}

/**
 * Draws left (y) and bottom (x) tick-marked axes, faint gridlines at each
 * major tick, and axis unit labels, into the plot area implied by the
 * caller's own toX/toY closures. Callers should draw this BEFORE their data
 * lines, so gridlines sit behind the plotted curves.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} opts
 * @param {(v:number)=>number} opts.toX
 * @param {(v:number)=>number} opts.toY
 * @param {number} opts.xMin
 * @param {number} opts.xMax
 * @param {number} opts.yMin
 * @param {number} opts.yMax
 * @param {number} [opts.yMinorStep] tick-mark-only interval for the y-axis, anchored at 0
 * @param {number} [opts.yMajorStep] labeled+gridlined interval for the y-axis (must be a multiple of yMinorStep)
 * @param {number} opts.plotLeft left edge of the plot area (start of the x-axis line)
 * @param {number} opts.plotRight right edge of the plot area (end of gridlines)
 * @param {number} opts.plotBottom bottom edge of the plot area (the x-axis line itself)
 * @param {number} opts.plotTop top edge of the plot area (end of gridlines)
 * @param {string} [opts.xLabel]
 * @param {string} [opts.yLabel]
 */
export function drawAxes(ctx, { toX, toY, xMin, xMax, yMin, yMax, yMinorStep, yMajorStep, plotLeft, plotRight, plotBottom, plotTop, xLabel = "", yLabel = "" }) {
  const xTicks = niceTicks(xMin, xMax);
  const yTicks = yMinorStep ? stepTicks(yMin, yMax, yMinorStep) : niceTicks(yMin, yMax);

  ctx.save();
  ctx.font = "10px monospace";
  ctx.lineWidth = 1;

  // y-axis: gridline + label only on major ticks; minor ticks get a short
  // tick mark with no gridline/label, so a fine step (e.g. every 1 or 5
  // units) doesn't turn into visual clutter.
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of yTicks) {
    const y = toY(v);
    const major = !yMinorStep || !yMajorStep || isMajorTick(v, yMajorStep);

    if (major) {
      ctx.strokeStyle = GRID_LINE_COLOR;
      ctx.beginPath();
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(plotRight, y);
      ctx.stroke();
    }

    ctx.strokeStyle = AXIS_LINE_COLOR;
    ctx.beginPath();
    ctx.moveTo(plotLeft - (major ? TICK_LENGTH : MINOR_TICK_LENGTH), y);
    ctx.lineTo(plotLeft, y);
    ctx.stroke();

    if (major) {
      ctx.fillStyle = LABEL_COLOR;
      const label = yMinorStep ? formatStepTick(v, yMajorStep ?? yMinorStep) : formatTick(v, yMax - yMin);
      ctx.fillText(label, plotLeft - TICK_LENGTH - 3, y);
    }
  }

  // Gridlines + x-axis ticks/labels
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of xTicks) {
    const x = toX(v);
    ctx.strokeStyle = GRID_LINE_COLOR;
    ctx.beginPath();
    ctx.moveTo(x, plotTop);
    ctx.lineTo(x, plotBottom);
    ctx.stroke();

    ctx.strokeStyle = AXIS_LINE_COLOR;
    ctx.beginPath();
    ctx.moveTo(x, plotBottom);
    ctx.lineTo(x, plotBottom + TICK_LENGTH);
    ctx.stroke();

    ctx.fillStyle = LABEL_COLOR;
    // Fixed at tenths, not formatTick's span-adaptive decimals -- this axis
    // is always time in seconds, and hundredths read as distracting noise
    // as the rolling window slides (see rollingWindow.js).
    ctx.fillText(v.toFixed(1), x, plotBottom + TICK_LENGTH + 1);
  }

  // Axis lines themselves
  ctx.strokeStyle = AXIS_LINE_COLOR;
  ctx.beginPath();
  ctx.moveTo(plotLeft, plotTop);
  ctx.lineTo(plotLeft, plotBottom);
  ctx.lineTo(plotRight, plotBottom);
  ctx.stroke();

  // Unit labels -- drawn in their own rows/columns, clear of the tick
  // numbers and of each panel's title (which now lives outside the panel
  // entirely -- see index.html's .panel-label).
  ctx.fillStyle = LABEL_COLOR;
  if (xLabel) {
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(xLabel, plotRight, plotBottom + TICK_LENGTH + 14);
  }
  if (yLabel) {
    ctx.save();
    ctx.translate(8, (plotTop + plotBottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }

  ctx.restore();
}
