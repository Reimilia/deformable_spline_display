import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { quaternionToMatrix } from "../scenes/RacketScene.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Shared with the Python physics tests -- same oracle, same tolerance basis.
// See physics/tests/test_rigid_body.py::test_quat_to_R_matches_reference.
const fixturePath = path.join(
  __dirname,
  "../../../physics/tests/fixtures/quaternion_rotation_pairs.json"
);
const cases = JSON.parse(readFileSync(fixturePath, "utf-8"));

/** THREE.Matrix4.elements is column-major; return the 3x3 rotation part, row-major. */
function matrix4ToRowMajor3x3(m) {
  const e = m.elements;
  return [
    [e[0], e[4], e[8]],
    [e[1], e[5], e[9]],
    [e[2], e[6], e[10]],
  ];
}

describe("quaternionToMatrix matches the Python quat_to_R oracle", () => {
  for (const { name, q, R } of cases) {
    it(name, () => {
      const m = quaternionToMatrix(q);
      const got = matrix4ToRowMajor3x3(m);
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          expect(got[i][j]).toBeCloseTo(R[i][j], 9);
        }
      }
    });
  }
});
