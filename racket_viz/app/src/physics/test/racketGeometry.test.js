import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { buildRacket } from "../racketGeometry.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function camelParams(p) {
  return {
    handleLength: p.handle_length,
    hoopLength: p.hoop_length,
    hoopWidth: p.hoop_width,
    totalMass: p.total_mass,
    hoopFraction: p.hoop_fraction,
  };
}

describe("buildRacket", () => {
  it("has I1 < I2 < I3 and orthonormal, proper-rotation principal axes", () => {
    const result = buildRacket();
    const [I1, I2, I3] = result.I;
    expect(I1).toBeLessThan(I2);
    expect(I2).toBeLessThan(I3);

    const evecs = result.evecs;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let dot = 0;
        for (let k = 0; k < 3; k++) dot += evecs[k][i] * evecs[k][j];
        expect(dot).toBeCloseTo(i === j ? 1 : 0, 6);
      }
    }
  });

  it("partitions vertices between handle and hoop with no gaps/overlaps", () => {
    const result = buildRacket();
    const n = result.vertsBody.length;
    const allIdx = [...result.handleIdx, ...result.hoopIdx].sort((a, b) => a - b);
    expect(allIdx).toEqual(Array.from({ length: n }, (_, i) => i));
  });

  it("identifies imin/imid/imax as a permutation of {0,1,2}", () => {
    const result = buildRacket();
    expect(new Set([result.imin, result.imid, result.imax])).toEqual(new Set([0, 1, 2]));
  });

  it("face_normal_body is a unit vector aligned with the imax axis", () => {
    // Guaranteed exactly (not approximately) by the perpendicular axis
    // theorem, since the racket is planar -- see racket_geometry.py's
    // build_racket docstring and the matching Python test.
    const result = buildRacket();
    const n = result.faceNormalBody;
    const mag = Math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2);
    expect(mag).toBeCloseTo(1.0, 9);

    const expected = [0, 0, 0];
    expected[result.imax] = 1;
    const matchesPositive = expected.every((v, i) => Math.abs(n[i] - v) < 1e-9);
    const matchesNegative = expected.every((v, i) => Math.abs(n[i] + v) < 1e-9);
    expect(matchesPositive || matchesNegative).toBe(true);
  });

  it("comUnweighted (unlike com) is invariant to hoopFraction and totalMass, only to shape", () => {
    // Regression for a real bug: main.js's live geometry preview used `com`
    // (mass-weighted) to recenter the racket mesh, so dragging hoopFraction
    // alone -- a pure mass-ratio parameter -- visibly translated the racket
    // on screen even though its shape hadn't changed. comUnweighted is the
    // stable anchor the preview uses instead.
    const base = buildRacket({ hoopFraction: 0.6, totalMass: 1.0 });
    for (const params of [
      { hoopFraction: 0.2, totalMass: 1.0 },
      { hoopFraction: 0.8, totalMass: 1.0 },
      { hoopFraction: 0.6, totalMass: 2.5 },
    ]) {
      const result = buildRacket(params);
      for (let i = 0; i < 3; i++) {
        expect(result.comUnweighted[i]).toBeCloseTo(base.comUnweighted[i], 9);
      }
    }
    // Sanity check the bug is real: `com` DOES move with hoopFraction.
    const other = buildRacket({ hoopFraction: 0.2 });
    expect(Math.abs(other.com[0] - base.com[0])).toBeGreaterThan(0.1);
  });

  it("matches Python's racket_geometry.build_racket() eigenvalues across several geometries", () => {
    const fixturePath = path.join(__dirname, "../../../../physics/tests/fixtures/racket_geometry_cases.json");
    const cases = JSON.parse(readFileSync(fixturePath, "utf-8"));

    for (const { params, I } of cases) {
      const result = buildRacket(camelParams(params));
      for (let i = 0; i < 3; i++) {
        expect(result.I[i]).toBeCloseTo(I[i], 6);
      }
    }
  });
});
