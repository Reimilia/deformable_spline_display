import { describe, expect, it } from "vitest";

import { eigh3x3 } from "../eigen3x3.js";

function matVec(A, v) {
  return A.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
}

function transpose3(A) {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
}

function matMul3(A, B) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return out;
}

/** A@v ≈ λv for every computed eigenpair -- doesn't assume a particular sign
 * or exact vector, which eigenvectors aren't unique up to anyway. */
function assertIsValidEigendecomposition(A, { values, vectors }, tol = 1e-9) {
  expect(values[0]).toBeLessThanOrEqual(values[1] + 1e-9);
  expect(values[1]).toBeLessThanOrEqual(values[2] + 1e-9);

  for (let k = 0; k < 3; k++) {
    const v = [vectors[0][k], vectors[1][k], vectors[2][k]];
    const Av = matVec(A, v);
    for (let i = 0; i < 3; i++) {
      expect(Av[i]).toBeCloseTo(values[k] * v[i], 6);
    }
  }

  const VtV = matMul3(transpose3(vectors), vectors);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      expect(VtV[i][j]).toBeCloseTo(i === j ? 1 : 0, tol < 1e-8 ? 8 : 6);
    }
  }

  const det =
    vectors[0][0] * (vectors[1][1] * vectors[2][2] - vectors[1][2] * vectors[2][1]) -
    vectors[0][1] * (vectors[1][0] * vectors[2][2] - vectors[1][2] * vectors[2][0]) +
    vectors[0][2] * (vectors[1][0] * vectors[2][1] - vectors[1][1] * vectors[2][0]);
  expect(det).toBeCloseTo(1, 6);
}

describe("eigh3x3", () => {
  it("handles an already-diagonal matrix", () => {
    const A = [
      [0.075, 0, 0],
      [0, 0.875, 0],
      [0, 0, 0.95],
    ];
    const result = eigh3x3(A);
    expect(result.values.map((v) => Number(v.toFixed(6)))).toEqual([0.075, 0.875, 0.95]);
    assertIsValidEigendecomposition(A, result);
  });

  it("handles a general symmetric matrix built from a known rotation", () => {
    // A = R @ diag(1,2,3) @ R^T for a 45-degree rotation about z -- has
    // genuine off-diagonal terms, unlike any matrix this project's racket
    // geometry actually produces (the racket is always exactly diagonal by
    // planar/mirror symmetry -- see racketGeometry.test.js). This is the
    // only test that actually exercises the general rotated case.
    const c = Math.SQRT1_2;
    const R = [
      [c, -c, 0],
      [c, c, 0],
      [0, 0, 1],
    ];
    const D = [
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3],
    ];
    const A = matMul3(matMul3(R, D), transpose3(R));

    const result = eigh3x3(A);
    expect(result.values[0]).toBeCloseTo(1, 6);
    expect(result.values[1]).toBeCloseTo(2, 6);
    expect(result.values[2]).toBeCloseTo(3, 6);
    assertIsValidEigendecomposition(A, result);
  });

  it("handles a near-degenerate matrix (two close eigenvalues)", () => {
    // The analytic cubic-root method has a known edge case near repeated
    // roots (small discriminant); confirm it stays numerically well-behaved.
    const c = Math.SQRT1_2;
    const R = [
      [c, -c, 0],
      [c, c, 0],
      [0, 0, 1],
    ];
    const D = [
      [2, 0, 0],
      [0, 2.0001, 0],
      [0, 0, 5],
    ];
    const A = matMul3(matMul3(R, D), transpose3(R));

    const result = eigh3x3(A);
    expect(result.values[0]).toBeCloseTo(2, 3);
    expect(result.values[1]).toBeCloseTo(2.0001, 3);
    expect(result.values[2]).toBeCloseTo(5, 3);
    assertIsValidEigendecomposition(A, result, 1e-6);
  });
});
