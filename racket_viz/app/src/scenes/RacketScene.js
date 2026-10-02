import * as THREE from "three";
import { THEME } from "../theme.js";

/**
 * Converts a scalar-first quaternion [w, x, y, z] (the convention produced by
 * physics/rigid_body.py) into a THREE.Matrix4 rotation.
 *
 * This is the single most important conversion in the app: a scalar-first vs.
 * scalar-last or handedness mismatch here silently produces a "looks physically
 * wrong" racket, not a crash -- see RacketScene.orientation.test.js.
 *
 * THREE.Quaternion's constructor is scalar-LAST (x, y, z, w), so the only real
 * work here is the reindex; THREE itself builds the rotation matrix.
 *
 * @param {[number, number, number, number]} q [q0, q1, q2, q3], q0 = scalar part
 * @returns {THREE.Matrix4}
 */
export function quaternionToMatrix(q) {
  const [q0, q1, q2, q3] = q;
  const threeQuat = new THREE.Quaternion(q1, q2, q3, q0);
  threeQuat.normalize();
  const m = new THREE.Matrix4();
  m.makeRotationFromQuaternion(threeQuat);
  return m;
}

/**
 * Builds the racket mesh once, in the body frame (handle line, hoop outline,
 * two colored faces offset by +/- face_thickness along face_normal_body).
 * Offsetting in body-frame local space (rather than recomputing a world-frame
 * offset every frame, as the matplotlib reference does) is equivalent for a
 * rigid rotation -- R(v + t*n) = Rv + t*Rn -- and lets the whole mesh be
 * rotated per-frame with a single group.quaternion assignment instead of
 * rebuilding geometry every frame.
 *
 * @param {object} geometry the scenario's `geometry` block (see ARCHITECTURE.md)
 * @returns {THREE.Group}
 */
export function createRacketMesh(geometry) {
  const { verts_body: verts, handle_idx: handleIdx, hoop_idx: hoopIdx, face_thickness: thickness, face_normal_body } = geometry;
  const normal = new THREE.Vector3(...face_normal_body);

  const group = new THREE.Group();

  // WebGL ignores LineBasicMaterial's `linewidth` on almost every browser/GPU
  // (a long-standing spec quirk -- lines always render at 1px regardless of
  // the value set), so a THREE.Line here would never look thicker no matter
  // what we set. Using a thin TubeGeometry instead gives real, visible
  // thickness -- purely cosmetic, the physics still treats the handle/hoop as
  // idealized 1D curves regardless of how thick we draw them.
  const TUBE_RADIUS = 0.025;

  const handlePts = handleIdx.map((i) => new THREE.Vector3(...verts[i]));
  const handleCurve = new THREE.CatmullRomCurve3(handlePts, false);
  const handleMesh = new THREE.Mesh(
    new THREE.TubeGeometry(handleCurve, 20, TUBE_RADIUS, 8, false),
    new THREE.MeshBasicMaterial({ color: THEME.racketTube })
  );
  group.add(handleMesh);

  const hoopPts = hoopIdx.map((i) => new THREE.Vector3(...verts[i]));
  const hoopCurve = new THREE.CatmullRomCurve3(hoopPts, true);
  const hoopMesh = new THREE.Mesh(
    new THREE.TubeGeometry(hoopCurve, 64, TUBE_RADIUS, 8, true),
    new THREE.MeshBasicMaterial({ color: THEME.racketTube })
  );
  group.add(hoopMesh);

  function buildFace(sign, color) {
    const n = hoopIdx.length;
    const positions = new Float32Array((n + 1) * 3);
    const center = new THREE.Vector3();
    hoopPts.forEach((p) => center.add(p));
    center.divideScalar(n).addScaledVector(normal, sign * thickness);
    positions.set([center.x, center.y, center.z], 0);

    hoopPts.forEach((p, k) => {
      const offset = p.clone().addScaledVector(normal, sign * thickness);
      positions.set([offset.x, offset.y, offset.z], (k + 1) * 3);
    });

    const indices = [];
    for (let k = 0; k < n; k++) {
      indices.push(0, k + 1, ((k + 1) % n) + 1);
    }

    const faceGeom = new THREE.BufferGeometry();
    faceGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    faceGeom.setIndex(indices);
    faceGeom.computeVertexNormals();

    return new THREE.Mesh(
      faceGeom,
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.85 })
    );
  }

  group.add(buildFace(1, THEME.racketFaceA));
  group.add(buildFace(-1, THEME.racketFaceB));

  return group;
}

/** Applies a scalar-first quaternion frame to an already-built racket group. */
export function updateRacketOrientation(group, q) {
  group.quaternion.setFromRotationMatrix(quaternionToMatrix(q));
}
