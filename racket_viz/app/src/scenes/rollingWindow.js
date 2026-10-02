/**
 * Computes the visible time window for a scrolling ("oscilloscope-style")
 * time-series panel: shows only the last `windowSeconds` of data up to the
 * current frame, never future data.
 *
 * This is what makes a long run (e.g. the app's 300s default) legible: without
 * it, a time-series panel maps the *entire* run's duration onto one fixed
 * panel width, so most of it renders as an unreadably compressed smear.
 *
 * The window itself is always exactly `windowSeconds` wide, from the very
 * first frame -- it does NOT grow from empty. Growing the window instead
 * (mapping [t0, currentT] to the full width while currentT < windowSeconds)
 * keeps changing the effective seconds-per-pixel scale every frame, which
 * visibly "compresses" already-drawn data leftward as playback starts. With a
 * fixed-size window from t0, the scale never changes: during this initial
 * "ramp-up" phase (currentT - t0 < windowSeconds) the window's right edge sits
 * ahead of the data, so callers that draw a "now" marker at `toX(t[currentIndex])`
 * naturally see it slide in from the left and reach the right edge exactly
 * when ramp-up ends -- do NOT hardcode the marker to the panel's right edge,
 * or it will appear pinned there throughout ramp-up instead of moving.
 * From that point on the window slides continuously, "now" pinned at the
 * right edge, matching the pre-existing behavior exactly.
 *
 * @param {number[]} t frame timestamps, strictly increasing
 * @param {number} currentIndex
 * @param {number} windowSeconds
 * @returns {{ startIdx: number, endIdx: number, windowStart: number, windowEnd: number }}
 */
export function computeVisibleWindow(t, currentIndex, windowSeconds) {
  const currentT = t[currentIndex];
  const windowEnd = Math.max(t[0] + windowSeconds, currentT);
  const windowStart = windowEnd - windowSeconds;

  let startIdx = 0;
  while (startIdx < currentIndex && t[startIdx] < windowStart) startIdx++;

  return { startIdx, endIdx: currentIndex, windowStart, windowEnd };
}
