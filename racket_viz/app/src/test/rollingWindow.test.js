import { describe, expect, it } from "vitest";

import { computeVisibleWindow } from "../scenes/rollingWindow.js";

// t = [0, 1, 2, ..., 19], one sample per second for a simple index<->time mapping.
const t = Array.from({ length: 20 }, (_, i) => i);

describe("computeVisibleWindow", () => {
  it("uses a fixed-size window from t0, even before the run is as old as the window", () => {
    // Regression: windowEnd used to equal currentT during this "ramp-up"
    // phase, which kept changing the seconds-per-pixel scale every frame
    // (visible as data "compressing" leftward). It must stay pinned at
    // t0+windowSeconds instead, so the scale never changes.
    const { startIdx, endIdx, windowStart, windowEnd } = computeVisibleWindow(t, 3, 10);
    expect(startIdx).toBe(0);
    expect(endIdx).toBe(3);
    expect(windowStart).toBe(0);
    expect(windowEnd).toBe(10);
  });

  it("slides once current time exceeds the window width, pinning 'now' at the end", () => {
    const { startIdx, endIdx, windowStart, windowEnd } = computeVisibleWindow(t, 15, 10);
    expect(windowEnd).toBe(15);
    expect(windowStart).toBe(5);
    expect(endIdx).toBe(15);
    expect(t[startIdx]).toBeGreaterThanOrEqual(5);
    expect(t[startIdx - 1] ?? -Infinity).toBeLessThan(5);
  });

  it("never shows a window wider than windowSeconds, during ramp-up or sliding", () => {
    for (const idx of [0, 3, 9, 10, 15, 19]) {
      const { windowStart, windowEnd } = computeVisibleWindow(t, idx, 10);
      expect(windowEnd - windowStart).toBeCloseTo(10, 9);
    }
  });

  it("handles the very first frame with the window already at full (fixed) size", () => {
    const { startIdx, endIdx, windowStart, windowEnd } = computeVisibleWindow(t, 0, 10);
    expect(startIdx).toBe(0);
    expect(endIdx).toBe(0);
    expect(windowStart).toBe(0);
    expect(windowEnd).toBe(10);
  });

  it("places 'now' (t[currentIndex]) at a position that advances during ramp-up, reaching the right edge exactly when ramp-up ends", () => {
    const toFraction = (idx) => {
      const { windowStart, windowEnd } = computeVisibleWindow(t, idx, 10);
      return (t[idx] - windowStart) / (windowEnd - windowStart);
    };
    expect(toFraction(0)).toBeCloseTo(0, 9);
    expect(toFraction(3)).toBeCloseTo(0.3, 9);
    expect(toFraction(9)).toBeCloseTo(0.9, 9);
    expect(toFraction(10)).toBeCloseTo(1, 9);
    expect(toFraction(15)).toBeCloseTo(1, 9); // sliding: 'now' stays pinned at the edge
  });
});
