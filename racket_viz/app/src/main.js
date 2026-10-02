import * as THREE from "three";

import { validateScenario } from "./core/DataLoader.js";
import { Playback } from "./core/Playback.js";
import { createRacketMesh, updateRacketOrientation } from "./scenes/RacketScene.js";
import { createSphereScene, updateDot } from "./scenes/SphereScene.js";
import { computeOmega, computeOmegaDomain, drawTimeSeries } from "./scenes/TimeSeriesPanel.js";
import {
  computeH,
  referencesForScenario,
  drawEnergyPanel,
  createHamiltonianScene,
  updateEnergyEllipsoid,
  computeAutoFitDistance,
} from "./scenes/ControlPanel.js";
import {
  isAchievable,
  computeCumulativeWork,
  computeTorqueMagnitude,
  computeAchievedIndex,
  drawTorquePanel,
} from "./scenes/ControlMetrics.js";
import { buildScenario, DEFAULT_SPIN_RATE } from "./physics/scenarioRunner.js";
import { buildRacket } from "./physics/racketGeometry.js";
import { casimirRadius, buildBackgroundWithCurrentOrbit } from "./physics/phasePortrait.js";
import { THEME } from "./theme.js";

const racketCanvas = document.getElementById("racket-canvas");
const sphereCanvas = document.getElementById("sphere-canvas");
const omegaCanvas = document.getElementById("omega-canvas");
const omegaCtx = omegaCanvas.getContext("2d");
const controlCanvas = document.getElementById("control-canvas");
const controlCtx = controlCanvas.getContext("2d");
const control3dCanvas = document.getElementById("control-3d-canvas");
const playPauseBtn = document.getElementById("pause-button");
const timeReadout = document.getElementById("time-readout");
const racketLabel = document.getElementById("racket-label");
const controlPanelLabel = document.getElementById("control-panel-label");
const controlPanelLegend = document.getElementById("control-panel-legend");

// Duration is no longer a sidebar control (didn't make sense as one) -- runs
// default to a long fixed duration instead, stopped manually via the Stop
// button. Also a deliberate stress test: long integrations are exactly where
// we found the fixed-step RK4 drift bug in the background curves (see
// DECISIONS.md); running the actual played trajectory this long too helps
// surface whether integrateFull needs the same fix.
const DEFAULT_DURATION_SECONDS = 300;

// ── Sidebar elements ────────────────────────────────────────────────────────
const sliderIds = ["handleLength", "hoopLength", "hoopWidth", "totalMass", "hoopFraction", "windStd"];
const sliders = Object.fromEntries(sliderIds.map((id) => [id, document.getElementById(id)]));
const sliderVals = Object.fromEntries(sliderIds.map((id) => [id, document.getElementById(`${id}-val`)]));
const startAxisSelect = document.getElementById("startAxis");
const startingPerturbationSlider = document.getElementById("startingPerturbation");
const startingPerturbationVal = document.getElementById("startingPerturbation-val");
const endAxisSelect = document.getElementById("endAxis");
const controlOnCheckbox = document.getElementById("controlOn");
const windOnCheckbox = document.getElementById("windOn");
const controlSubfields = document.getElementById("control-subfields");
const windSubfields = document.getElementById("wind-subfields");
const runButton = document.getElementById("run-button");
const resetButton = document.getElementById("reset-button");
const hamiltonianViewToggle = document.getElementById("hamiltonianViewToggle");
const inertiaReadout = document.getElementById("inertia-readout");
const controlReadoutDiv = document.getElementById("control-readout");
const controlMetricsText = document.getElementById("control-metrics-text");
const torqueCanvas = document.getElementById("torque-canvas");
const torqueCtx = torqueCanvas.getContext("2d");

function refreshSliderLabels() {
  for (const id of sliderIds) {
    sliderVals[id].textContent = Number(sliders[id].value).toFixed(2);
  }
}
// Separate from the generic sliderIds above -- its 0.002 step needs 3 decimal
// places to actually show, not the 2 the geometry sliders use.
function refreshStartingPerturbationLabel() {
  startingPerturbationVal.textContent = Number(startingPerturbationSlider.value).toFixed(3);
}
function refreshSubfieldVisibility() {
  controlSubfields.style.display = controlOnCheckbox.checked ? "flex" : "none";
  windSubfields.style.display = windOnCheckbox.checked ? "flex" : "none";
  controlReadoutDiv.style.display = controlOnCheckbox.checked ? "flex" : "none";
  // torque-canvas is sized from clientWidth/Height, which is 0 while its
  // parent is display:none -- re-measure now that it's actually visible.
  if (controlOnCheckbox.checked) resize2DCanvas(torqueCanvas, torqueCtx);
}

function formatInertia(I) {
  // I is always sorted ascending -- I[0]=imin (smallest moment of inertia,
  // the LONG/handle axis), I[1]=imid (intermediate), I[2]=imax (largest
  // moment of inertia, the SHORT/face-normal axis) -- see racketGeometry.js.
  // Counterintuitive but correct: moment of inertia is smallest about an
  // object's longest axis (mass sits close to it) and largest about its
  // shortest axis (mass swings far around it).
  const rows = [
    ["I₁ (long axis)", I[0]],
    ["I₂ (intermediate)", I[1]],
    ["I₃ (short axis)", I[2]],
  ];
  const rowsHtml = rows
    .map(
      ([label, value]) =>
        `<div style="display:flex;justify-content:space-between;">${label}<span>${value.toFixed(4)}</span></div>`
    )
    .join("");
  return `<div style="margin-bottom:4px;">Moments of inertia</div>${rowsHtml}`;
}

function renderLegend(container, items) {
  container.innerHTML = items
    .map(({ color, label }) => `<span class="legend-item"><span class="legend-dot" style="background:${color}"></span>${label}</span>`)
    .join("");
}
for (const id of sliderIds) sliders[id].addEventListener("input", refreshSliderLabels);
startingPerturbationSlider.addEventListener("input", refreshStartingPerturbationLabel);
controlOnCheckbox.addEventListener("change", refreshSubfieldVisibility);
windOnCheckbox.addEventListener("change", refreshSubfieldVisibility);
refreshSliderLabels();
refreshStartingPerturbationLabel();
refreshSubfieldVisibility();

// "Explain" flip-card toggles: one generic listener for all 4 panels rather
// than four near-identical ones. Purely a CSS transform on the target
// .flip-inner -- the canvas underneath keeps rendering every frame
// regardless of which face is currently showing, same as how the 3D
// Hamiltonian view already keeps drawing while hidden.
document.querySelectorAll(".explain-switch").forEach((toggle) => {
  const flipTarget = document.getElementById(toggle.dataset.flip);
  toggle.addEventListener("change", () => {
    flipTarget.classList.toggle("flipped", toggle.checked);
  });
});

function readParamsFromSidebar() {
  return {
    handleLength: Number(sliders.handleLength.value),
    hoopLength: Number(sliders.hoopLength.value),
    hoopWidth: Number(sliders.hoopWidth.value),
    totalMass: Number(sliders.totalMass.value),
    hoopFraction: Number(sliders.hoopFraction.value),
    startAxis: startAxisSelect.value,
    startingPerturbation: Number(startingPerturbationSlider.value),
    controlOn: controlOnCheckbox.checked,
    endAxis: endAxisSelect.value,
    windOn: windOnCheckbox.checked,
    windStd: Number(sliders.windStd.value),
    T: DEFAULT_DURATION_SECONDS,
    N: Math.round(DEFAULT_DURATION_SECONDS * 100),
  };
}

function applyPreset(preset) {
  sliders.handleLength.value = preset.handleLength ?? 1.0;
  sliders.hoopLength.value = preset.hoopLength ?? 2.0;
  sliders.hoopWidth.value = preset.hoopWidth ?? 1.0;
  sliders.totalMass.value = preset.totalMass ?? 1.0;
  sliders.hoopFraction.value = preset.hoopFraction ?? 0.6;
  // preset.T is currently unused -- duration is no longer sidebar-controlled,
  // see DEFAULT_DURATION_SECONDS above. Left on preset objects below in case
  // presets (and their own duration) come back.
  startAxisSelect.value = preset.startAxis;
  startingPerturbationSlider.value = preset.startingPerturbation ?? 0.02;
  controlOnCheckbox.checked = preset.controlOn;
  endAxisSelect.value = preset.endAxis ?? "imax";
  windOnCheckbox.checked = preset.windOn ?? false;
  sliders.windStd.value = preset.windStd ?? 0.05;
  refreshSliderLabels();
  refreshStartingPerturbationLabel();
  refreshSubfieldVisibility();
  runFromSidebar(preset.label, preset.caption);
}

// Preset buttons temporarily removed from index.html (not intuitive) -- kept
// here, commented out, to restore later rather than rebuild from scratch.
// T values below are the exact ones calibrated in physics/tests/test_controllers.py
// and app/src/physics/test/controllers.test.js (STAGE_D_T=6.0, Stage F T=8.0) --
// not arbitrary guesses.
//
// document.getElementById("preset-free").addEventListener("click", () =>
//   applyPreset({
//     startAxis: "imid",
//     controlOn: false,
//     windOn: false,
//     T: 20,
//     label: "Free spin",
//     caption: "Close to the tipping point, the racket suddenly flips over.",
//   })
// );
// document.getElementById("preset-imin").addEventListener("click", () =>
//   applyPreset({
//     startAxis: "imid",
//     controlOn: true,
//     endAxis: "imin",
//     windOn: false,
//     T: 6,
//     label: "Control on",
//     caption: "An active controller steers the spin toward axis imin.",
//   })
// );
// document.getElementById("preset-imax").addEventListener("click", () =>
//   applyPreset({
//     startAxis: "imid",
//     controlOn: true,
//     endAxis: "imax",
//     windOn: false,
//     T: 6,
//     label: "Control on",
//     caption: "An active controller steers the spin toward axis imax.",
//   })
// );
// document.getElementById("preset-align").addEventListener("click", () =>
//   applyPreset({
//     startAxis: "imid",
//     controlOn: true,
//     endAxis: "imax",
//     windOn: false,
//     T: 8,
//     label: "Alignment",
//     caption: "The controller brings both orientation and spin to rest at the target.",
//   })
// );

// ── Three.js setup ───────────────────────────────────────────────────────────
function makeRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  return renderer;
}

const racketRenderer = makeRenderer(racketCanvas);
const sphereRenderer = makeRenderer(sphereCanvas);
const controlRenderer = makeRenderer(control3dCanvas);

const racketScene = new THREE.Scene();
const sphereScene = new THREE.Scene();
const controlScene = new THREE.Scene();

// Match the 2D canvas panels' own background (drawn per-frame via
// ctx.fillStyle = THEME.panelBg) instead of THREE's default black clear --
// otherwise the WebGL panels read as a different, jarring background color.
const PANEL_BG = new THREE.Color(THEME.panelBg);
racketScene.background = PANEL_BG;
sphereScene.background = PANEL_BG;
controlScene.background = PANEL_BG;

const racketCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
racketCamera.position.set(2.2, 1.6, 2.6);
racketCamera.lookAt(0, 0, 0);

// A controlled run's M(t) is not conserved, so the dot can travel far off
// the unit-radius background (fixed points/curves, all built from the same
// R_cas) -- most extremely when start and end axes have very different
// moments of inertia. A fixed camera then leaves the dot clipped outside the
// frame entirely (looks like it vanished). Auto-fit like controlCamera below.
const sphereCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
const SPHERE_CAMERA_INITIAL_POS = new THREE.Vector3(1.8, 1.4, 2.2);
const SPHERE_CAMERA_DIR = SPHERE_CAMERA_INITIAL_POS.clone().normalize();
sphereCamera.position.copy(SPHERE_CAMERA_INITIAL_POS);
sphereCamera.lookAt(0, 0, 0);

// Matches sphereCamera's own framing -- both panels look at the same kind of
// M-space content, so a consistent viewing angle makes them easy to compare.
const controlCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
const CONTROL_CAMERA_INITIAL_POS = new THREE.Vector3(1.8, 1.4, 2.2);
const CONTROL_CAMERA_DIR = CONTROL_CAMERA_INITIAL_POS.clone().normalize();
controlCamera.position.copy(CONTROL_CAMERA_INITIAL_POS);
controlCamera.lookAt(0, 0, 0);

let racketGroup = null;
let sphereDot = null;
let controlDot = null;
let currentEllipsoid = null;
let desiredEllipsoid = null;
// Smoothed, not snapped straight to target -- the ellipsoid's own scale can
// jump around a bit frame to frame (wind is noisy), and a camera that
// zooms in lockstep with every jitter would be uncomfortable to watch.
let controlCameraDistance = CONTROL_CAMERA_INITIAL_POS.length();
let sphereCameraDistance = SPHERE_CAMERA_INITIAL_POS.length();
let playback = null;
let currentDoc = null;
let omegaSeries = null;
let omegaDomain = null;
let hSeries = null;
let hReferences = null;
let torqueMagnitudeSeries = null;
let cumulativeWorkSeries = null;
let achievedIndex = null;
let lastTorqueTextMs = null;

function resize2DCanvas(canvas, ctx) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resize() {
  for (const [renderer, camera, canvas] of [
    [racketRenderer, racketCamera, racketCanvas],
    [sphereRenderer, sphereCamera, sphereCanvas],
    [controlRenderer, controlCamera, control3dCanvas],
  ]) {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  resize2DCanvas(omegaCanvas, omegaCtx);
  resize2DCanvas(controlCanvas, controlCtx);
  resize2DCanvas(torqueCanvas, torqueCtx);
}
window.addEventListener("resize", resize);

function clearScene(scene) {
  while (scene.children.length) scene.remove(scene.children[0]);
}

/** Rebuilds just the racket *shape* (not a trajectory) from the current
 * geometry sliders, so the mesh updates live while dragging -- no simulation
 * involved, so this needs no server and is cheap enough to run on every
 * "input" event. Preserves whatever orientation is currently displayed.
 * totalMass is deliberately excluded: it cancels out of the center-of-mass
 * math entirely and never changes vertex positions. */
function previewRacketGeometry() {
  const geometry = buildRacket({
    handleLength: Number(sliders.handleLength.value),
    hoopLength: Number(sliders.hoopLength.value),
    hoopWidth: Number(sliders.hoopWidth.value),
    hoopFraction: Number(sliders.hoopFraction.value),
  });
  const previousQuaternion = racketGroup ? racketGroup.quaternion.clone() : null;

  // Display-only fix: `vertsBody` is centered on the mass-weighted COM, which
  // shifts along x as hoopFraction alone changes -- correct for the real
  // simulation (rotation must be about the true COM), but for this static,
  // non-rotating preview it just makes the racket visibly slide around while
  // dragging a mass-ratio slider that isn't supposed to affect its shape.
  // Re-anchor on the mass-independent geometric centroid instead, purely for
  // this preview; the actual Run path below (renderDoc) is untouched.
  const [dx, dy, dz] = geometry.com.map((c, i) => c - geometry.comUnweighted[i]);
  const evecs = geometry.evecs;
  const offsetBody = [
    dx * evecs[0][0] + dy * evecs[1][0] + dz * evecs[2][0],
    dx * evecs[0][1] + dy * evecs[1][1] + dz * evecs[2][1],
    dx * evecs[0][2] + dy * evecs[1][2] + dz * evecs[2][2],
  ];
  const vertsDisplay = geometry.vertsBody.map((v) => [
    v[0] + offsetBody[0],
    v[1] + offsetBody[1],
    v[2] + offsetBody[2],
  ]);

  clearScene(racketScene);
  racketGroup = createRacketMesh({
    verts_body: vertsDisplay,
    handle_idx: geometry.handleIdx,
    hoop_idx: geometry.hoopIdx,
    face_thickness: geometry.faceThickness,
    face_normal_body: geometry.faceNormalBody,
  });
  if (previousQuaternion) racketGroup.quaternion.copy(previousQuaternion);
  racketScene.add(racketGroup);

  inertiaReadout.innerHTML = formatInertia(geometry.I);
}
for (const id of ["handleLength", "hoopLength", "hoopWidth", "hoopFraction"]) {
  sliders[id].addEventListener("input", previewRacketGeometry);
}

/** Rebuilds just the sphere panel's background (wireframe sphere, trajectory/
 * separatrix curves, fixed points, and the current near-axis orbit curve) so
 * it reflects the current Starting axis / Starting perturbation / geometry
 * sliders live, without needing Run -- mirrors previewRacketGeometry above,
 * but for the sphere panel. Does not touch the moving dot/ellipsoid or any
 * currently-playing scenario; those only update again on the next Run, same
 * as the racket shape preview already does for the racket panel. */
function previewSphereBackground() {
  const geometry = buildRacket({
    handleLength: Number(sliders.handleLength.value),
    hoopLength: Number(sliders.hoopLength.value),
    hoopWidth: Number(sliders.hoopWidth.value),
    hoopFraction: Number(sliders.hoopFraction.value),
  });
  const { I, imin, imid, imax } = geometry;
  const axisIndex = { imin, imid, imax };
  const startAxisIdx = axisIndex[startAxisSelect.value];
  // Matches buildScenario's own R_cas rule exactly (see scenarioRunner.js):
  // free runs conserve |M0|, which for imin/imax can differ hugely from
  // I[imid]*spinRate, so the background scales off the actual start axis;
  // controlled runs scale off the actual TARGET (endAxis) instead, since
  // that's what the controller actually drives |M| toward.
  const R_cas = casimirRadius(
    I,
    controlOnCheckbox.checked ? axisIndex[endAxisSelect.value] : startAxisIdx,
    DEFAULT_SPIN_RATE
  );
  const kickFraction = Number(startingPerturbationSlider.value);

  const background = buildBackgroundWithCurrentOrbit(
    I,
    R_cas,
    imin,
    imax,
    imid,
    startAxisIdx,
    DEFAULT_SPIN_RATE,
    kickFraction
  );

  clearScene(sphereScene);
  const { group, dot } = createSphereScene(
    {
      trajectories: background.trajectories,
      separatrices: background.separatrices,
      stable_fixed_points: background.stableFixedPoints,
      unstable_fixed_points: background.unstableFixedPoints,
    },
    R_cas
  );
  sphereScene.add(group);
  sphereDot = dot;
}
for (const id of ["handleLength", "hoopLength", "hoopWidth", "hoopFraction"]) {
  sliders[id].addEventListener("input", previewSphereBackground);
}
startAxisSelect.addEventListener("change", previewSphereBackground);
startingPerturbationSlider.addEventListener("input", previewSphereBackground);
controlOnCheckbox.addEventListener("change", previewSphereBackground);
endAxisSelect.addEventListener("change", previewSphereBackground);
previewSphereBackground();

/** Swaps panel 4 between its flat 2D chart and the 3D ellipsoid view. */
function refreshHamiltonianView() {
  const show3d = hamiltonianViewToggle.checked;
  controlCanvas.style.display = show3d ? "none" : "block";
  control3dCanvas.style.display = show3d ? "block" : "none";
  // control3dCanvas is sized from clientWidth/Height, which is 0 while
  // display:none -- re-measure now that it's actually visible (same fix as
  // torque-canvas's own display:none -> visible transition).
  if (show3d) {
    const w = control3dCanvas.clientWidth;
    const h = control3dCanvas.clientHeight;
    controlRenderer.setSize(w, h, false);
    controlCamera.aspect = w / h;
    controlCamera.updateProjectionMatrix();
  }
}
hamiltonianViewToggle.addEventListener("change", refreshHamiltonianView);
refreshHamiltonianView();

function segmentAt(index) {
  const segs = currentDoc.meta.segments;
  return segs.find((s) => index >= s.start && index < s.end) ?? segs[segs.length - 1];
}

function updateFrame(index) {
  const q = currentDoc.frames.quaternion[index];
  const M = currentDoc.frames.M_body[index];
  updateRacketOrientation(racketGroup, q);
  updateDot(sphereDot, M, currentDoc.meta.R_cas);

  // Auto-fit: a controlled run's |M(t)| is not conserved, so the dot can sit
  // well off the unit-radius background (fixed points/curves, all built from
  // the same R_cas) -- most extremely when the start and end axes have very
  // different moments of inertia. Without this, a fixed camera leaves the
  // dot clipped outside the frame (looks like it disappeared). Smoothed for
  // the same reason as controlCamera below.
  const sphereMaxExtent = Math.max(1, sphereDot.position.length());
  const sphereTargetDistance = computeAutoFitDistance(sphereMaxExtent, sphereCamera.fov);
  sphereCameraDistance += (sphereTargetDistance - sphereCameraDistance) * 0.1;
  sphereCamera.position.copy(SPHERE_CAMERA_DIR).multiplyScalar(sphereCameraDistance);
  sphereCamera.lookAt(0, 0, 0);

  // Panel 4's 3D view: kept live regardless of whether it's the currently
  // shown view, same as the other WebGL scenes render every frame
  // regardless of which 2D/3D panels happen to be visible right now.
  updateDot(controlDot, M, currentDoc.meta.R_cas);
  updateEnergyEllipsoid(currentEllipsoid, hSeries[index], currentDoc.meta.I, currentDoc.meta.R_cas);

  // Auto-fit: the ellipsoid isn't bounded to stay near the reference
  // sphere's radius 1 the way the sphere panel's own content always is (see
  // computeAutoFitDistance's docstring) -- keep the camera far enough back
  // to frame whichever of {reference sphere, both ellipsoids, the dot} is
  // currently largest, smoothed so it doesn't jump/jitter frame to frame.
  const maxExtent = Math.max(
    1,
    currentEllipsoid.scale.x,
    currentEllipsoid.scale.y,
    currentEllipsoid.scale.z,
    desiredEllipsoid.visible ? Math.max(desiredEllipsoid.scale.x, desiredEllipsoid.scale.y, desiredEllipsoid.scale.z) : 0,
    controlDot.position.length()
  );
  const targetDistance = computeAutoFitDistance(maxExtent, controlCamera.fov);
  controlCameraDistance += (targetDistance - controlCameraDistance) * 0.1;
  controlCamera.position.copy(CONTROL_CAMERA_DIR).multiplyScalar(controlCameraDistance);
  controlCamera.lookAt(0, 0, 0);

  const seg = segmentAt(index);
  racketLabel.textContent = seg.label;
  timeReadout.textContent = `t = ${currentDoc.frames.t[index].toFixed(1)}s`;

  drawTimeSeries(omegaCtx, {
    t: currentDoc.frames.t,
    omega: omegaSeries,
    domain: omegaDomain,
    currentIndex: index,
    width: omegaCanvas.clientWidth,
    height: omegaCanvas.clientHeight,
  });

  drawEnergyPanel(controlCtx, {
    t: currentDoc.frames.t,
    H: hSeries,
    references: hReferences,
    currentIndex: index,
    width: controlCanvas.clientWidth,
    height: controlCanvas.clientHeight,
  });

  if (currentDoc.meta.mode === "controlled") {
    drawTorquePanel(torqueCtx, {
      t: currentDoc.frames.t,
      torqueMagnitude: torqueMagnitudeSeries,
      currentIndex: index,
      width: torqueCanvas.clientWidth,
      height: torqueCanvas.clientHeight,
    });
    updateControlMetricsText(index);
  }
}

/** "Current torque" is throttled to roughly twice a second -- per-frame
 * updates would just flicker unreadably, since the controller's raw output
 * can vary quickly. "Work done thus far" is throttled along with it for one
 * self-consistent readout rather than two numbers updating at different
 * rates side by side. */
function updateControlMetricsText(index) {
  const nowMs = performance.now();
  if (lastTorqueTextMs !== null && nowMs - lastTorqueTextMs < 500) return;
  lastTorqueTextMs = nowMs;

  const workSoFar = cumulativeWorkSeries[index];
  const currentTorque = torqueMagnitudeSeries[index];
  let html = `Work done thus far: ${workSoFar.toFixed(3)} J<br>Current torque: ${currentTorque.toFixed(3)} N&middot;m`;
  if (achievedIndex !== null && index >= achievedIndex) {
    html += `<br><span class="done-message">Done! Achieved at t = ${currentDoc.frames.t[achievedIndex].toFixed(2)}s</span>`;
  }
  controlMetricsText.innerHTML = html;
}

/** Renders a scenario doc that's already been built (by scenarioRunner.js,
 * entirely client-side -- see ARCHITECTURE.md). Validated against the same
 * schema fetched static JSON used to satisfy, as a safety net, even though
 * nothing is fetched anymore. */
function renderDoc(doc) {
  validateScenario(doc);
  currentDoc = doc;

  clearScene(racketScene);
  clearScene(sphereScene);
  clearScene(controlScene);

  racketGroup = createRacketMesh(doc.geometry);
  racketScene.add(racketGroup);

  const { group: sphereGroup, dot } = createSphereScene(doc.background, doc.meta.R_cas);
  sphereScene.add(sphereGroup);
  sphereDot = dot;
  // Reset for the same reason as controlCameraDistance below -- don't carry
  // over a previous run's zoom level into this one's frame-0 state.
  sphereCameraDistance = SPHERE_CAMERA_INITIAL_POS.length();

  const {
    group: hamiltonianGroup,
    currentEllipsoid: newCurrentEllipsoid,
    desiredEllipsoid: newDesiredEllipsoid,
    dot: newControlDot,
  } = createHamiltonianScene();
  controlScene.add(hamiltonianGroup);
  currentEllipsoid = newCurrentEllipsoid;
  desiredEllipsoid = newDesiredEllipsoid;
  controlDot = newControlDot;
  // Reset the auto-fit camera to its neutral default rather than carrying
  // over a previous run's (possibly very different) distance -- updateFrame
  // will smoothly zoom from here to this run's actual frame-0 target.
  controlCameraDistance = CONTROL_CAMERA_INITIAL_POS.length();
  // Fixed for the whole run (unlike currentEllipsoid, updated every frame in
  // updateFrame) -- a controlled target doesn't change once the run starts.
  desiredEllipsoid.visible = doc.meta.mode === "controlled";
  if (doc.meta.mode === "controlled") {
    updateEnergyEllipsoid(desiredEllipsoid, doc.meta.desired_H, doc.meta.I, doc.meta.R_cas);
    controlPanelLabel.textContent = "Current vs. desired (H)";
  } else {
    controlPanelLabel.textContent = "Hamiltonian energy (H)";
  }

  omegaSeries = computeOmega(doc.frames.M_body, doc.meta.I);
  omegaDomain = computeOmegaDomain(omegaSeries);

  hSeries = computeH(doc.frames.M_body, doc.meta.I);
  hReferences = referencesForScenario(doc.meta);
  renderLegend(controlPanelLegend, [{ color: "#eeeeee", label: "current" }, ...hReferences]);

  torqueMagnitudeSeries = computeTorqueMagnitude(doc.frames.controller_torque);
  cumulativeWorkSeries = computeCumulativeWork(doc.frames.controller_torque, doc.frames.M_body, doc.meta.I, doc.frames.t);
  achievedIndex = computeAchievedIndex(doc.frames.M_body, doc.frames.t, doc.meta);
  lastTorqueTextMs = null;

  inertiaReadout.innerHTML = formatInertia(doc.meta.I);

  playback = new Playback(doc.frames.t);
  updateFrame(0);
  playback.play();
  playPauseBtn.textContent = "Pause";
}

function runFromSidebar(label, caption) {
  const params = readParamsFromSidebar();
  if (label !== undefined) params.label = label;
  if (caption !== undefined) params.caption = caption;

  // Correct physics (holding a controller's target exactly where it already
  // started produces no motion), but indistinguishable from a stuck/broken
  // run without saying so explicitly.
  if (params.controlOn && params.startAxis === params.endAxis && label === undefined) {
    params.label = "Already at target";
    params.caption = "Starting and ending axis are the same -- holding steady, nothing to correct.";
  }

  const doc = buildScenario(params);
  renderDoc(doc);
}

let lastFrameMs = null;

function animate(nowMs) {
  requestAnimationFrame(animate);

  const dt = lastFrameMs === null ? 0 : (nowMs - lastFrameMs) / 1000;
  lastFrameMs = nowMs;

  if (playback) {
    if (playback.playing) {
      const alive = playback.tick(dt);
      if (!alive) playPauseBtn.textContent = "Replay";
    }
    updateFrame(playback.frameIndexAtTime(playback.currentTime));
  }

  racketRenderer.render(racketScene, racketCamera);
  sphereRenderer.render(sphereScene, sphereCamera);
  controlRenderer.render(controlScene, controlCamera);
}

playPauseBtn.addEventListener("click", () => {
  if (!playback) return;
  if (playback.playing) {
    playback.pause();
    playPauseBtn.textContent = "Play";
    return;
  }
  if (playPauseBtn.textContent === "Replay") {
    playback.seek(0);
  }
  playback.play();
  playPauseBtn.textContent = "Pause";
});

runButton.addEventListener("click", () => runFromSidebar());

// Distinct from Pause: Reset halts AND rewinds to the beginning, rather than
// just freezing wherever playback currently is -- Pause is what preserves
// the current view (readouts, graphs) for a closer look.
resetButton.addEventListener("click", () => {
  if (!playback) return;
  playback.pause();
  playback.seek(0);
  playPauseBtn.textContent = "Play";
  updateFrame(playback.frameIndexAtTime(playback.currentTime));
});

resize();
applyPreset({
  startAxis: "imid",
  controlOn: false,
  windOn: false,
  T: 20,
  label: "Free spin",
  caption: "Close to the tipping point, the racket suddenly flips over.",
});
requestAnimationFrame(animate);
