import { describe, expect, it } from "vitest";

import { computeOmega, computeOmegaDomain } from "../scenes/TimeSeriesPanel.js";

describe("computeOmega", () => {
  it("divides each M_body component by the corresponding inertia", () => {
    const M_body = [
      [1.5, 0.0, 0.0],
      [0.0, 0.875, 0.0],
    ];
    const I = [0.075, 0.875, 0.95];
    const omega = computeOmega(M_body, I);
    expect(omega[0]).toEqual([20, 0, 0]);
    expect(omega[1]).toEqual([0, 1, 0]);
  });
});

describe("computeOmegaDomain", () => {
  it("returns a padded [min, max] spanning every component", () => {
    const omega = [
      [-1, 5, 2],
      [3, -2, 0],
    ];
    const [min, max] = computeOmegaDomain(omega, 0.1);
    // raw range is [-2, 5], padded by 10% of the 7-wide range (0.7) each side
    expect(min).toBeCloseTo(-2.7, 6);
    expect(max).toBeCloseTo(5.7, 6);
  });

  it("does not collapse to a zero-width domain for a constant series", () => {
    const omega = [
      [1, 1, 1],
      [1, 1, 1],
    ];
    const [min, max] = computeOmegaDomain(omega);
    expect(max).toBeGreaterThan(min);
  });
});
