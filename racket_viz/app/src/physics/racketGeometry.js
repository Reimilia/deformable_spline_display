/**
 * Tennis-racket geometry -> mass distribution -> principal-axis inertia tensor.
 * Ported from physics/racket_geometry.py, using eigen3x3.js in place of
 * numpy.linalg.eigh (which already returns ascending eigenvalues with a
 * proper-rotation eigenvector matrix, so no separate det-flip step is needed
 * here -- eigh3x3 handles it internally).
 */
import { eigh3x3 } from "./eigen3x3.js";

/**
 * @param {object} [params]
 * @param {number} [params.handleLength=1.0]
 * @param {number} [params.hoopLength=2.0]
 * @param {number} [params.hoopWidth=1.0]
 * @param {number} [params.totalMass=1.0]
 * @param {number} [params.hoopFraction=0.60]
 * @param {number} [params.faceThickness=0.04]
 * @param {number} [params.numHandlePts=40]
 * @param {number} [params.numHoopPts=120]
 * @returns {{
 *   vertsBody: number[][], handleIdx: number[], hoopIdx: number[],
 *   I: number[], evecs: number[][], imin: number, imid: number, imax: number,
 *   faceThickness: number, faceNormalBody: number[], com: number[],
 *   comUnweighted: number[]
 * }}
 */
export function buildRacket(params = {}) {
  const {
    handleLength = 1.0,
    hoopLength = 2.0,
    hoopWidth = 1.0,
    totalMass = 1.0,
    hoopFraction = 0.6,
    faceThickness = 0.04,
    numHandlePts = 40,
    numHoopPts = 120,
  } = params;

  const handleFraction = 1.0 - hoopFraction;
  const hoopRadiusX = hoopLength / 2;
  const hoopRadiusY = hoopWidth / 2;
  const hoopCenterX = hoopRadiusX;

  const handlePts = [];
  for (let i = 0; i < numHandlePts; i++) {
    const x = -handleLength + (handleLength * i) / (numHandlePts - 1);
    handlePts.push([x, 0, 0]);
  }

  const hoopPts = [];
  for (let i = 0; i < numHoopPts; i++) {
    const theta = (2 * Math.PI * i) / numHoopPts;
    hoopPts.push([hoopCenterX + hoopRadiusX * Math.cos(theta), hoopRadiusY * Math.sin(theta), 0]);
  }

  const verticesRaw = [...handlePts, ...hoopPts];
  const handleIdx = Array.from({ length: numHandlePts }, (_, i) => i);
  const hoopIdx = Array.from({ length: numHoopPts }, (_, i) => i + numHandlePts);

  const masses = new Array(verticesRaw.length);
  const handleMassEach = (handleFraction * totalMass) / numHandlePts;
  const hoopMassEach = (hoopFraction * totalMass) / numHoopPts;
  for (const i of handleIdx) masses[i] = handleMassEach;
  for (const i of hoopIdx) masses[i] = hoopMassEach;

  const com = [0, 0, 0];
  for (let i = 0; i < verticesRaw.length; i++) {
    com[0] += verticesRaw[i][0] * masses[i];
    com[1] += verticesRaw[i][1] * masses[i];
    com[2] += verticesRaw[i][2] * masses[i];
  }
  com[0] /= totalMass;
  com[1] /= totalMass;
  com[2] /= totalMass;

  // Unweighted (purely geometric) centroid -- unlike `com`, this depends only
  // on the shape (handleLength/hoopLength/hoopWidth), never on hoopFraction
  // or totalMass, so it's a stable anchor for a live shape-only preview (see
  // main.js's previewRacketGeometry): `com` shifts along x by ~1 full unit as
  // hoopFraction alone is dragged across its range, which visibly translates
  // the racket on screen even though its actual shape hasn't changed.
  const comUnweighted = [0, 0, 0];
  for (const v of verticesRaw) {
    comUnweighted[0] += v[0];
    comUnweighted[1] += v[1];
    comUnweighted[2] += v[2];
  }
  comUnweighted[0] /= verticesRaw.length;
  comUnweighted[1] /= verticesRaw.length;
  comUnweighted[2] /= verticesRaw.length;

  const vertsC = verticesRaw.map((v) => [v[0] - com[0], v[1] - com[1], v[2] - com[2]]);

  const Itensor = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < vertsC.length; i++) {
    const r = vertsC[i];
    const m = masses[i];
    const r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        Itensor[a][b] += m * ((a === b ? r2 : 0) - r[a] * r[b]);
      }
    }
  }

  const { values: I, vectors: evecs } = eigh3x3(Itensor);

  const vertsBody = vertsC.map((r) => [
    r[0] * evecs[0][0] + r[1] * evecs[1][0] + r[2] * evecs[2][0],
    r[0] * evecs[0][1] + r[1] * evecs[1][1] + r[2] * evecs[2][1],
    r[0] * evecs[0][2] + r[1] * evecs[1][2] + r[2] * evecs[2][2],
  ]);

  // face_normal_body = evecs^T @ [0,0,1] = the 3rd ROW of evecs (columns are
  // eigenvectors, so evecs^T's 3rd row is evecs's 3rd row transposed back)
  const rawNormal = [evecs[2][0], evecs[2][1], evecs[2][2]];
  const normalMag = Math.sqrt(rawNormal[0] ** 2 + rawNormal[1] ** 2 + rawNormal[2] ** 2);
  const faceNormalBody = rawNormal.map((x) => x / normalMag);

  return {
    vertsBody,
    handleIdx,
    hoopIdx,
    I,
    evecs,
    imin: 0,
    imid: 1,
    imax: 2,
    faceThickness,
    faceNormalBody,
    com,
    comUnweighted,
  };
}
