import * as THREE from "three";
import { THEME } from "../theme.js";

/**
 * @param {[number, number, number]} M_body angular momentum in body frame
 * @param {number} R_cas Casimir radius at scenario start (for normalizing the
 *   display sphere to unit radius -- note a *controlled* scenario's |M(t)| can
 *   move away from this value; the dot is drawn at its true (possibly
 *   off-sphere) position, not clamped back onto it)
 * @returns {[number, number, number]} position for the moving dot, in the same
 *   normalized units as the static sphere mesh
 */
export function dotPosition(M_body, R_cas) {
  const [m1, m2, m3] = M_body;
  return [m1 / R_cas, m2 / R_cas, m3 / R_cas];
}

/**
 * Builds the static sphere panel background once: wireframe unit sphere,
 * trajectory/separatrix curves, and stable/unstable fixed points -- all
 * normalized by the scenario's initial R_cas so the sphere itself is unit
 * radius. Returns the group plus the moving dot mesh (updated every frame via
 * `updateDot`).
 *
 * @param {object} background the scenario's `background` block
 * @param {number} R_cas
 * @returns {{ group: THREE.Group, dot: THREE.Mesh }}
 */
export function createSphereScene(background, R_cas) {
  const group = new THREE.Group();

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(1, 32, 24),
    new THREE.MeshBasicMaterial({ color: THEME.referenceSphere, wireframe: true, transparent: true, opacity: 0.15 })
  );
  group.add(sphere);

  // Background curves lie almost exactly at |M|/R_cas == 1 (the same radius
  // as the sphere mesh itself), which z-fights against the sphere surface --
  // classic coincident-depth WebGL artifact, showing up as speckled/broken
  // lines rather than the smooth curves the underlying (well-converged,
  // 900-point) integration actually produces. Pushing the curves out to a
  // hair beyond the sphere's radius resolves it with no visible gap.
  const CURVE_RADIUS_SCALE = 1.006;

  function toPoints(curve) {
    return curve.map(
      ([x, y, z]) =>
        new THREE.Vector3(
          (x / R_cas) * CURVE_RADIUS_SCALE,
          (y / R_cas) * CURVE_RADIUS_SCALE,
          (z / R_cas) * CURVE_RADIUS_SCALE
        )
    );
  }

  function addCurves(curves, color) {
    for (const curve of curves) {
      const geom = new THREE.BufferGeometry().setFromPoints(toPoints(curve));
      group.add(new THREE.Line(geom, new THREE.LineBasicMaterial({ color })));
    }
  }
  addCurves(background.trajectories, THEME.trajectory);

  // The separatrix curves are drawn as thin tubes, not THREE.Line, for the
  // same reason RacketScene.js's handle/hoop are: WebGL ignores
  // LineBasicMaterial's `linewidth` on nearly every browser/GPU, so a 1px
  // line is all a Line ever renders regardless of styling. That's especially
  // a problem here because these curves are exact analytic loops that can be
  // very "flat" (their excursion off the [imid,imax] plane, sqrt((I3-I2)*I1 /
  // ((I2-I1)*I3)) as a fraction of R_cas, is often small) -- a 1px line at
  // that scale all but disappears against the sphere's own wireframe grid
  // (confirmed empirically: correct, present pixels that are nevertheless
  // essentially invisible in a screenshot). A tube gives real, visible girth
  // regardless of how flat the loop is.
  const SEPARATRIX_TUBE_RADIUS = 0.01;
  for (const curve of background.separatrices) {
    const pts = toPoints(curve);
    const tubeCurve = new THREE.CatmullRomCurve3(pts, false);
    const mesh = new THREE.Mesh(
      new THREE.TubeGeometry(tubeCurve, pts.length, SEPARATRIX_TUBE_RADIUS, 6, false),
      new THREE.MeshBasicMaterial({ color: THEME.separatrix })
    );
    group.add(mesh);
  }

  function addPoints(points, color) {
    for (const [x, y, z] of points) {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.035, 12, 12),
        new THREE.MeshBasicMaterial({ color })
      );
      mesh.position.set(x / R_cas, y / R_cas, z / R_cas);
      group.add(mesh);
    }
  }
  addPoints(background.stable_fixed_points, THEME.stable);
  addPoints(background.unstable_fixed_points, THEME.unstable);

  // Matches the fixed points' own radius (both stable and unstable use the
  // same 0.035) -- the moving dot shouldn't visually outrank the landmarks
  // it's moving between.
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.035, 16, 16),
    new THREE.MeshBasicMaterial({ color: THEME.current })
  );
  group.add(dot);

  return { group, dot };
}

/** Moves the sphere panel's dot mesh to the current frame's M_body. */
export function updateDot(dot, M_body, R_cas) {
  dot.position.set(...dotPosition(M_body, R_cas));
}
