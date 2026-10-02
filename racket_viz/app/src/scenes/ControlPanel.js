/**
 * Panel 4: current vs. desired state, with a 2D and a 3D view (toggled in the
 * UI). The 2D view (drawEnergyPanel) always plots the current H(t); which
 * reference lines get drawn against it depends on the scenario's mode (see
 * ARCHITECTURE.md):
 *   - free: the 3 natural equilibrium energies (meta.H_axis) -- H(t) is
 *     exactly conserved, so this shows which energy band the racket is in.
 *   - controlled: a single target (meta.desired_H) -- H(t) is NOT conserved
 *     here, so this is a literal current-vs-desired convergence plot.
 *
 * The 3D view (createHamiltonianScene) shows the same information spatially:
 * the energy level set {M : H(M) = const} is itself an ellipsoid in
 * angular-momentum space (semi-axes sqrt(2*H*I_k)) -- the sphere panel's own
 * background curves are exactly its intersection with the Casimir sphere.
 * This used to be an optional overlay ON the sphere panel; it's its own
 * panel now, so it always shows the "current" ellipsoid (constant-sized
 * exactly when H truly is conserved -- free, no wind; visibly resizing
 * otherwise, since wind or control both do real work on the racket) plus,
 * for controlled scenarios only, a second fixed "desired" ellipsoid at
 * meta.desired_H.
 *
 * Color convention: the free-mode references reuse panel 3's
 * stable(imin)/unstable(imid)/imax convention from theme.js; the
 * controlled-mode target uses THEME.target.
 */
import * as THREE from "three";
import { computeVisibleWindow } from "./rollingWindow.js";
import { drawAxes, computeNiceStep } from "./axisTicks.js";
import { THEME } from "../theme.js";

export const CONTROLLED_TARGET_COLOR = THEME.target;
const AXIS_COLORS = [THEME.stable, THEME.unstable, THEME.imax];

/**
 * @param {[number, number, number][]} M_body
 * @param {[number, number, number]} I
 * @returns {number[]} H(t) = 0.5 * sum(M_i^2 / I_i)
 */
export function computeH(M_body, I) {
  const [I1, I2, I3] = I;
  return M_body.map(([m1, m2, m3]) => 0.5 * (m1 * m1 / I1 + m2 * m2 / I2 + m3 * m3 / I3));
}

/**
 * @param {object} meta a scenario's `meta` block
 * @returns {{ value: number, color: string, label: string }[]}
 */
export function referencesForScenario(meta) {
  if (meta.mode === "free") {
    const [hMin, hMid, hMax] = meta.H_axis;
    return [
      { value: hMin, color: AXIS_COLORS[0], label: "H(long axis)" },
      { value: hMid, color: AXIS_COLORS[1], label: "H(intermediate axis)" },
      { value: hMax, color: AXIS_COLORS[2], label: "H(short axis)" },
    ];
  }
  return [{ value: meta.desired_H, color: CONTROLLED_TARGET_COLOR, label: "desired H" }];
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} opts
 * @param {number[]} opts.t
 * @param {number[]} opts.H
 * @param {{value:number,color:string,label:string}[]} opts.references
 * @param {number} opts.currentIndex
 * @param {number} opts.width
 * @param {number} opts.height
 * @param {number} [opts.windowSeconds=5]
 */
export function drawEnergyPanel(ctx, { t, H, references, currentIndex, width, height, windowSeconds = 5 }) {
  const values = [...H, ...references.map((r) => r.value)];
  let yMin = Math.min(...values);
  let yMax = Math.max(...values);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.1;
  // H = 0.5*sum(M_i^2/I_i) can never be negative -- floor at 0 (rather than
  // padding below it like the top) so "0" sits exactly at the plot's
  // bottom-left corner instead of leaving a dead strip of unreachable space
  // below a slightly-negative padded minimum.
  yMin = Math.max(0, yMin - pad);
  yMax += pad;

  const { startIdx, endIdx, windowStart, windowEnd } = computeVisibleWindow(t, currentIndex, windowSeconds);
  const marginLeft = 46;
  const marginRight = 18;
  const marginTop = 10;
  const marginBottom = 34;
  const plotW = width - marginLeft - marginRight;
  const plotH = height - marginBottom - marginTop;
  const plotRight = width - marginRight;
  const plotBottom = marginTop + plotH;
  const toX = (ti) => marginLeft + ((ti - windowStart) / (windowEnd - windowStart || 1)) * plotW;
  const toY = (v) => marginTop + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = THEME.panelBg;
  ctx.fillRect(0, 0, width, height);

  const { majorStep, minorStep } = computeNiceStep(yMax - yMin);
  drawAxes(ctx, {
    toX,
    toY,
    xMin: windowStart,
    xMax: windowEnd,
    yMin,
    yMax,
    yMinorStep: minorStep,
    yMajorStep: majorStep,
    plotLeft: marginLeft,
    plotRight,
    plotTop: marginTop,
    plotBottom,
    xLabel: "t (s)",
    yLabel: "Hamiltonian (H)",
  });

  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1.5;
  for (const ref of references) {
    ctx.strokeStyle = ref.color;
    const y = toY(ref.value);
    ctx.beginPath();
    ctx.moveTo(marginLeft, y);
    ctx.lineTo(plotRight, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  ctx.strokeStyle = THEME.text;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = startIdx; i <= endIdx; i++) {
    const x = toX(t[i]);
    const y = toY(H[i]);
    if (i === startIdx) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // "Now" marker at its actual x position, not hardcoded to the right edge --
  // see rollingWindow.js: it advances left-to-right during ramp-up and only
  // reaches the edge once the window starts sliding.
  ctx.fillStyle = THEME.current;
  ctx.beginPath();
  ctx.arc(toX(t[currentIndex]), toY(H[currentIndex]), 3.5, 0, 2 * Math.PI);
  ctx.fill();
}

/**
 * Builds panel 4's 3D scene: a faint reference sphere at the Casimir radius
 * (context -- this is where the initial |M|=R_cas sphere sits, so the
 * ellipsoid's own scale, normalized the same way, is directly comparable to
 * it), a "current" energy-level ellipsoid (always visible, live-updating --
 * see updateEnergyEllipsoid), a "desired" ellipsoid only meaningful for
 * controlled scenarios (fixed at meta.desired_H, hidden otherwise), and a dot
 * at the current M-position (same convention as the sphere panel's own dot,
 * reuse SphereScene.js's dotPosition/updateDot for it).
 * @returns {{ group: THREE.Group, currentEllipsoid: THREE.Mesh,
 *   desiredEllipsoid: THREE.Mesh, dot: THREE.Mesh }}
 */
export function createHamiltonianScene() {
  const group = new THREE.Group();

  const referenceSphere = new THREE.Mesh(
    new THREE.SphereGeometry(1, 24, 18),
    new THREE.MeshBasicMaterial({ color: THEME.referenceSphere, wireframe: true, transparent: true, opacity: 0.12 })
  );
  group.add(referenceSphere);

  const currentEllipsoid = new THREE.Mesh(
    new THREE.SphereGeometry(1, 24, 18),
    new THREE.MeshBasicMaterial({ color: THEME.text, wireframe: true, transparent: true, opacity: 0.55 })
  );
  group.add(currentEllipsoid);

  const desiredEllipsoid = new THREE.Mesh(
    new THREE.SphereGeometry(1, 24, 18),
    new THREE.MeshBasicMaterial({ color: CONTROLLED_TARGET_COLOR, wireframe: true, transparent: true, opacity: 0.35 })
  );
  desiredEllipsoid.visible = false;
  group.add(desiredEllipsoid);

  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.05, 16, 16),
    new THREE.MeshBasicMaterial({ color: THEME.current })
  );
  group.add(dot);

  return { group, currentEllipsoid, desiredEllipsoid, dot };
}

/**
 * Rescales an energy-ellipsoid mesh to a given H. Semi-axis along principal
 * axis k is sqrt(2*H*I_k), normalized by R_cas to match the sphere/dot's
 * units. Used for both the "current" ellipsoid (called every frame with
 * H(t)) and the "desired" ellipsoid (called once with meta.desired_H, since
 * a controlled target doesn't change over the run).
 * @param {THREE.Mesh} ellipsoid
 * @param {number} H
 * @param {number[]} I [I1,I2,I3]
 * @param {number} R_cas
 */
export function updateEnergyEllipsoid(ellipsoid, H, I, R_cas) {
  ellipsoid.scale.set(
    Math.sqrt(Math.max(0, 2 * H * I[0])) / R_cas,
    Math.sqrt(Math.max(0, 2 * H * I[1])) / R_cas,
    Math.sqrt(Math.max(0, 2 * H * I[2])) / R_cas
  );
}

/**
 * Distance a perspective camera (given its vertical fov, in degrees) needs
 * from the origin -- looking at the origin, along whatever direction it's
 * already pointed -- to fit an object of bounding radius `maxExtent` inside
 * the frame, with `margin` extra breathing room (1.0 would put it exactly
 * at the frame's edge).
 *
 * Needed because panel 4's reference sphere is always radius 1, but the
 * energy ellipsoid is NOT bounded by that at all: for a free run, R_cas is
 * now based on the actual starting axis (see scenarioRunner.js), so if that
 * axis's own moment of inertia is disproportionately small relative to the
 * other two (easily reached within the existing geometry slider ranges --
 * confirmed with all 4 mass/shape sliders at their minimums), R_cas becomes
 * tiny while the ellipsoid's OTHER semi-axes (governed by the larger
 * moments) don't shrink to match, so the ellipsoid can legitimately balloon
 * to several times the reference sphere's radius. That's correct physics
 * (H stays exactly conserved throughout -- verified numerically), not a
 * data bug -- but a fixed camera distance can't frame both the tame default
 * case and this several-times-larger case well. margin=1.3 is chosen so the
 * common maxExtent=1 case reproduces the original hardcoded framing
 * (distance ~3.17) closely, rather than visibly jumping for the typical case.
 *
 * @param {number} maxExtent
 * @param {number} fovDegrees
 * @param {number} [margin=1.3]
 * @returns {number}
 */
export function computeAutoFitDistance(maxExtent, fovDegrees, margin = 1.3) {
  const halfFovRad = ((fovDegrees / 2) * Math.PI) / 180;
  return (Math.max(maxExtent, 1e-6) * margin) / Math.tan(halfFovRad);
}
