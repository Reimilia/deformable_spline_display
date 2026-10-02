import { describe, expect, it } from "vitest";

import { niceTicks, formatTick, stepTicks, isMajorTick, formatStepTick, computeNiceStep } from "../scenes/axisTicks.js";

describe("niceTicks", () => {
  it("returns `count` evenly-spaced values inclusive of min and max", () => {
    const ticks = niceTicks(0, 8, 5);
    expect(ticks).toEqual([0, 2, 4, 6, 8]);
  });

  it("defaults to 5 ticks", () => {
    expect(niceTicks(0, 4)).toHaveLength(5);
  });

  it("does not divide by zero for a collapsed [min, max]", () => {
    expect(niceTicks(3, 3, 5)).toEqual([3]);
  });

  it("handles negative ranges", () => {
    const ticks = niceTicks(-10, 10, 5);
    expect(ticks).toEqual([-10, -5, 0, 5, 10]);
  });
});

describe("formatTick", () => {
  it("uses more decimals for a small span", () => {
    expect(formatTick(0.125, 0.5)).toBe("0.125");
  });

  it("uses fewer decimals for a large span", () => {
    expect(formatTick(12.4, 100)).toBe("12");
  });

  it("never returns more than 3 decimal places", () => {
    expect(formatTick(0.00012, 0.0005)).toMatch(/^-?\d+\.\d{0,3}$/);
  });

  it("falls back to a fixed format for a zero span", () => {
    expect(formatTick(5, 0)).toBe("5.00");
  });
});

describe("stepTicks", () => {
  it("is anchored at 0, not at min", () => {
    // omega panel case: domain like [-7.6, 8.1], step 1 -- ticks must land on
    // whole units (...,-1,0,1,...), not on an arbitrary -7.6-relative offset.
    const ticks = stepTicks(-7.6, 8.1, 1);
    expect(ticks[0]).toBe(-7);
    expect(ticks).toContain(0);
    expect(ticks[ticks.length - 1]).toBe(8);
  });

  it("covers a wider step (energy panel case)", () => {
    const ticks = stepTicks(-3, 53, 5);
    expect(ticks).toEqual([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]);
  });

  it("returns an empty array for a non-positive step", () => {
    expect(stepTicks(0, 10, 0)).toEqual([]);
  });
});

describe("isMajorTick", () => {
  it("flags multiples of the major step", () => {
    expect(isMajorTick(4, 2)).toBe(true);
    expect(isMajorTick(0, 2)).toBe(true);
    expect(isMajorTick(-2, 2)).toBe(true);
  });

  it("does not flag values between major ticks", () => {
    expect(isMajorTick(3, 2)).toBe(false);
    expect(isMajorTick(1, 2)).toBe(false);
  });
});

describe("formatStepTick", () => {
  it("prints whole-number steps with no decimals", () => {
    expect(formatStepTick(10, 10)).toBe("10");
    expect(formatStepTick(2, 2)).toBe("2");
    expect(formatStepTick(0, 2)).toBe("0");
  });

  it("prints fractional steps with matching decimals", () => {
    expect(formatStepTick(0.5, 0.5)).toBe("0.5");
  });
});

describe("computeNiceStep", () => {
  it("reproduces the omega panel's previously hand-tuned default (range ~16 -> major 2, minor 1)", () => {
    expect(computeNiceStep(16)).toEqual({ majorStep: 2, minorStep: 1 });
  });

  it("reproduces the energy panel's previously hand-tuned default (range ~220 -> major 20, minor 10)", () => {
    expect(computeNiceStep(220)).toEqual({ majorStep: 20, minorStep: 10 });
  });

  it("regression: a small range (e.g. H maxing out around 14) no longer collapses to a single visible tick", () => {
    // Previously a fixed majorStep=20 meant the only gridline below yMax=14
    // was "0" -- nothing else fit on the chart at all.
    const { majorStep } = computeNiceStep(14);
    expect(majorStep).toBeLessThan(14);
    expect(majorStep).toBe(2);
  });

  it("scales down for a very small range", () => {
    expect(computeNiceStep(3)).toEqual({ majorStep: 0.5, minorStep: 0.1 });
  });

  it("falls back to a safe default for a non-positive range", () => {
    expect(computeNiceStep(0)).toEqual({ majorStep: 1, minorStep: 0.5 });
  });
});
