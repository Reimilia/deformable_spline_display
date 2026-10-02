/**
 * Loads and validates a precomputed scenario JSON (see ARCHITECTURE.md for the
 * schema) exported by physics/export.py. Mirrors the validation in
 * physics/export.py::scenario_to_json so both sides reject the same shapes.
 */

const REQUIRED_META_KEYS = ["mode", "I", "R_cas", "segments"];
const REQUIRED_FRAME_KEYS = ["t", "quaternion", "M_body", "controller_torque"];
const REQUIRED_GEOMETRY_KEYS = [
  "verts_body", "handle_idx", "hoop_idx", "face_thickness", "face_normal_body",
];
const REQUIRED_BACKGROUND_KEYS = [
  "trajectories", "separatrices", "stable_fixed_points", "unstable_fixed_points",
];

/**
 * @param {object} doc parsed scenario JSON
 * @returns {object} the same doc, if valid
 * @throws {Error} with a specific message identifying what's wrong, if invalid
 */
export function validateScenario(doc) {
  if (!doc || typeof doc !== "object") {
    throw new Error("scenario must be an object");
  }
  const { meta, frames, geometry, background } = doc;
  if (!meta || typeof meta !== "object") {
    throw new Error("scenario missing 'meta'");
  }
  if (!frames || typeof frames !== "object") {
    throw new Error("scenario missing 'frames'");
  }
  if (!geometry || typeof geometry !== "object") {
    throw new Error("scenario missing 'geometry'");
  }
  if (!background || typeof background !== "object") {
    throw new Error("scenario missing 'background'");
  }

  const missingMeta = REQUIRED_META_KEYS.filter((k) => !(k in meta));
  if (missingMeta.length) {
    throw new Error(`scenario meta missing required keys: ${missingMeta.join(", ")}`);
  }
  if (meta.mode !== "free" && meta.mode !== "controlled") {
    throw new Error(`invalid mode: ${meta.mode}`);
  }
  if (meta.mode === "controlled" && (!("desired_H" in meta) || !("desired_L" in meta))) {
    throw new Error("controlled scenario meta missing desired_H/desired_L");
  }
  if (meta.mode === "free" && (!("H_axis" in meta) || meta.H_axis.length !== 3)) {
    throw new Error("free scenario meta missing H_axis (or wrong length)");
  }

  const missingFrames = REQUIRED_FRAME_KEYS.filter((k) => !(k in frames));
  if (missingFrames.length) {
    throw new Error(`scenario frames missing required keys: ${missingFrames.join(", ")}`);
  }

  const { t, quaternion, M_body: M, controller_torque: torque } = frames;
  const n = t.length;
  if (quaternion.length !== n || M.length !== n || torque.length !== n) {
    throw new Error(
      `frame array length mismatch: t=${n}, quaternion=${quaternion.length}, M_body=${M.length}, controller_torque=${torque.length}`
    );
  }
  for (const q of quaternion) {
    if (q.length !== 4) {
      throw new Error(`quaternion entry has length ${q.length}, expected 4`);
    }
  }
  for (const m of M) {
    if (m.length !== 3) {
      throw new Error(`M_body entry has length ${m.length}, expected 3`);
    }
  }
  for (const tau of torque) {
    if (tau.length !== 3) {
      throw new Error(`controller_torque entry has length ${tau.length}, expected 3`);
    }
  }

  const missingGeometry = REQUIRED_GEOMETRY_KEYS.filter((k) => !(k in geometry));
  if (missingGeometry.length) {
    throw new Error(`scenario geometry missing required keys: ${missingGeometry.join(", ")}`);
  }
  for (const v of geometry.verts_body) {
    if (v.length !== 3) {
      throw new Error(`geometry.verts_body entry has length ${v.length}, expected 3`);
    }
  }
  if (geometry.face_normal_body.length !== 3) {
    throw new Error("geometry.face_normal_body must have length 3");
  }

  const missingBackground = REQUIRED_BACKGROUND_KEYS.filter((k) => !(k in background));
  if (missingBackground.length) {
    throw new Error(`scenario background missing required keys: ${missingBackground.join(", ")}`);
  }
  for (const curve of [...background.trajectories, ...background.separatrices]) {
    if (!Array.isArray(curve)) {
      throw new Error("background trajectory/separatrix entries must be arrays of points");
    }
  }
  for (const p of [...background.stable_fixed_points, ...background.unstable_fixed_points]) {
    if (p.length !== 3) {
      throw new Error(`background fixed point has length ${p.length}, expected 3`);
    }
  }

  return doc;
}

/**
 * @param {string} url path to a scenario JSON file
 * @returns {Promise<object>} the validated scenario
 */
export async function loadScenario(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to load scenario at ${url}: ${response.status} ${response.statusText}`);
  }
  const doc = await response.json();
  return validateScenario(doc);
}
