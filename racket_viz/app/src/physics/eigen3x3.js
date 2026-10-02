/**
 * Closed-form eigendecomposition of a real symmetric 3x3 matrix.
 *
 * This is the one piece of the JS physics port with no direct Python
 * counterpart to translate line-by-line: `racket_geometry.py` leans on
 * `numpy.linalg.eigh` (LAPACK), which has no JS equivalent. Uses the standard
 * analytic method (characteristic cubic + trigonometric solution for three
 * real roots, which a real symmetric matrix always has) rather than an
 * iterative solver -- exact for the 3x3 case, no convergence concerns.
 *
 * See app/src/physics/test/eigen3x3.test.js for the correctness argument:
 * known diagonal matrices, a hand-built matrix with off-diagonal terms
 * verified via the A@v = lambda*v property (not exact eigenvector values,
 * which aren't unique up to sign), and a cross-check fixture against
 * Python's actual `racket_geometry.build_racket()` eigenvalues.
 */

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function norm(v) {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function det3(M) {
  return (
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
  );
}

/** Unit eigenvector for a known eigenvalue of symmetric A, via the standard
 * "cross product of two rows of (A - lambda*I)" trick: for a rank-2
 * (A - lambda*I), that cross product spans the 1-D null space. Falls back to
 * building a vector orthogonal to already-found eigenvectors only in the
 * (for us, practically unreachable) exactly-degenerate case. */
function eigenvectorFor(A, lambda, foundVecs) {
  const M = [
    [A[0][0] - lambda, A[0][1], A[0][2]],
    [A[1][0], A[1][1] - lambda, A[1][2]],
    [A[2][0], A[2][1], A[2][2] - lambda],
  ];
  const candidates = [cross(M[0], M[1]), cross(M[0], M[2]), cross(M[1], M[2])];

  let best = null;
  let bestNorm = -1;
  for (const c of candidates) {
    const n = norm(c);
    if (n > bestNorm) {
      bestNorm = n;
      best = c;
    }
  }

  if (bestNorm < 1e-8) {
    if (foundVecs.length === 0) {
      best = [1, 0, 0];
    } else if (foundVecs.length === 1) {
      const seed = Math.abs(foundVecs[0][0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      best = cross(foundVecs[0], seed);
    } else {
      best = cross(foundVecs[0], foundVecs[1]);
    }
  }

  const n = norm(best);
  return [best[0] / n, best[1] / n, best[2] / n];
}

/**
 * @param {number[][]} A 3x3 symmetric matrix (only the upper triangle is read)
 * @returns {{ values: number[], vectors: number[][] }} values ascending;
 *   vectors[k] is NOT the k-th eigenvector -- columns are, i.e. the
 *   eigenvector for values[k] is [vectors[0][k], vectors[1][k], vectors[2][k]].
 */
export function eigh3x3(A) {
  const a11 = A[0][0];
  const a22 = A[1][1];
  const a33 = A[2][2];
  const a12 = A[0][1];
  const a13 = A[0][2];
  const a23 = A[1][2];

  const p1 = a12 * a12 + a13 * a13 + a23 * a23;
  const scale = a11 * a11 + a22 * a22 + a33 * a33 + 1;

  let values;
  if (p1 < 1e-20 * scale) {
    values = [a11, a22, a33];
  } else {
    const q = (a11 + a22 + a33) / 3;
    const p2 = (a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2 * p1;
    const p = Math.sqrt(p2 / 6);

    const B = [
      [(a11 - q) / p, a12 / p, a13 / p],
      [a12 / p, (a22 - q) / p, a23 / p],
      [a13 / p, a23 / p, (a33 - q) / p],
    ];
    let r = det3(B) / 2;
    r = Math.max(-1, Math.min(1, r));
    const phi = r <= -1 ? Math.PI / 3 : r >= 1 ? 0 : Math.acos(r) / 3;

    const eig1 = q + 2 * p * Math.cos(phi); // largest
    const eig3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3); // smallest
    const eig2 = 3 * q - eig1 - eig3; // trace = eig1+eig2+eig3

    values = [eig1, eig2, eig3];
  }
  values.sort((a, b) => a - b);

  const foundVecs = [];
  for (const lambda of values) {
    foundVecs.push(eigenvectorFor(A, lambda, foundVecs));
  }

  const vectors = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let col = 0; col < 3; col++) {
    for (let row = 0; row < 3; row++) vectors[row][col] = foundVecs[col][row];
  }

  if (det3(vectors) < 0) {
    for (let row = 0; row < 3; row++) vectors[row][2] *= -1;
  }

  return { values, vectors };
}
