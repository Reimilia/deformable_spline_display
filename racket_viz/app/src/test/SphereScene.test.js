import { describe, expect, it } from "vitest";

import { dotPosition } from "../scenes/SphereScene.js";

describe("dotPosition", () => {
  it("normalizes a point on the initial Casimir sphere to the unit sphere", () => {
    const R_cas = 1.5;
    const [x, y, z] = dotPosition([1.5, 0, 0], R_cas);
    expect(x).toBeCloseTo(1.0, 9);
    expect(y).toBeCloseTo(0.0, 9);
    expect(z).toBeCloseTo(0.0, 9);
  });

  it("does not clamp a controlled-scenario point that has left the initial sphere", () => {
    // |M| = 2 * R_cas here -- a controlled trajectory can leave the initial
    // Casimir sphere (energy/momentum are not conserved under torque); the dot
    // must be drawn at its true position, not clamped back onto the unit sphere.
    const R_cas = 1.5;
    const [x, y, z] = dotPosition([3.0, 0, 0], R_cas);
    expect(x).toBeCloseTo(2.0, 9);
    expect(y).toBeCloseTo(0.0, 9);
    expect(z).toBeCloseTo(0.0, 9);
  });
});
