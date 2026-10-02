# Decisions log

## 2026-07-24 — Project kickoff, architecture, test-first scaffold

**Context.** Building a reusable four-panel Dzhanibekov-effect visualization
(racket 3D, Casimir sphere, angular velocity vs. time, current-vs-desired
control), styled after a greeting-card physics illustration. A prior attempt on
branch `feature/tennis-racket-visualizer` stalled; going slower and more
deliberately this time.

**Why the prior attempt likely failed.** It ran a live FastAPI + WebSocket
backend doing RK4 integration every frame, coupled to a loaded PyTorch NN
checkpoint (for an H_nn-vs-H_analytical comparison) and a full IDA-PBC
controller — a live research instrument and a public visualization at once. Left
untouched on its branch as historical reference; not deleted or merged.

**Decisions made:**
- Location: `LieSPHGP/racket_viz/`, new branch `feature/racket-viz` off `main`.
- Architecture: Python precomputes all trajectory/phase-portrait data once to
  static JSON; Three.js only loads and replays it. No live server, no
  WebSocket, no NN-checkpoint coupling at runtime.
- Tooling: pytest (physics), Vite + Vitest (app).
- Panel 4 reuses the *already-validated* analytic controllers from LieSPHGP's
  open PR #1 (`tennis-racket-effect` branch, `controller_stageD.py` /
  `controller_stageF.py`) rather than anything NN-based. Toggling "control on"
  + a target axis/alignment selects between a small number of *precomputed*
  controlled scenarios — no live simulation needed.
- Energy/angular-momentum conservation is a hard invariant only for **free**
  (torque-free) scenarios. Controlled scenarios must show `H(t)`/`|M(t)|`
  actually changing (the yellow dot leaving the initial Casimir sphere as the
  racket physically flips to the new target configuration) — this is flagged as
  the main new engineering work, not just glue code, and has its own explicit
  tests (`test_controlled_energy_is_not_conserved`,
  `test_controlled_angular_momentum_changes`) precisely so a bug that decouples
  the controller from the integrator doesn't silently look like the free case.
- Caption text is authored fresh per scenario/segment, not copied from
  `dzhanibekov_display.py`'s hardcoded `CAPTIONS_LEFT`/`CAPTIONS_RIGHT`.

**Process for this stage:** proposed the test list + architecture + docs plan,
got it approved, then wrote the full test suite (physics: 28 pytest cases across
5 files; app: 17 Vitest cases across 5 files) against stub modules that raise
`NotImplementedError` — confirmed red phase (all failing for the right reason,
no import/collection errors) before writing any real implementation. See
`ARCHITECTURE.md` for the data contract these tests assume.

**Known caveat in the current test suite**: a few `DataLoader.test.js` cases
assert `expect(() => validateScenario(bad)).toThrow()` — against the current
stub (which throws unconditionally) these pass "for free," without yet proving
the *right* validation logic fires. They'll need re-checking once
`validateScenario` is implemented (e.g. asserting on the error message) to make
sure they're still testing what they claim to.

**Next step:** implement `physics/` module bodies until `pytest` is green, then
`app/src/core` + `app/src/scenes` until `npm test` is green — in that order,
before any visual/styling work.

## 2026-07-24 — Physics implementation, `pytest` green (28/28)

Implemented `rigid_body.py`, `racket_geometry.py`, `phase_portrait.py`,
`controllers.py`, `scenarios.py`, `export.py` in that order (each was the direct
extraction/adaptation described in `ARCHITECTURE.md`). Two real issues surfaced
along the way, both fixed in the test file (not worked around in the
implementation) since the implementation was verified correct independently:

1. **`np.testing.assert_allclose` broadcasting quirk** (numpy 2.4.6): comparing
   a `(N, 3)` array against a `(3,)` array reports "not equal" even when every
   row is exactly identical — confirmed in isolation with a synthetic example,
   not a translation-layer bug. Fixed by using `np.allclose` (which broadcasts
   correctly) in `test_fixed_points_are_stationary` instead.

2. **The Dzhanibekov-flip test's original IC didn't actually flip** within any
   reasonable time. Root cause: the saddle at the unstable (imid) fixed point
   has a specific growing eigendirection — a fixed *ratio* between the
   perturbations on the two stable axes (≈11.6:1 for this geometry, derived
   from `sqrt((1/I_max−1/I_mid)/(1/I_mid−1/I_min))`), not a 1:1 split. The
   original test perturbed both stable axes by an equal amount *in M-space*,
   which landed almost exactly on the *decaying* eigendirection instead of the
   growing one, so the flip would technically still happen but on an
   impractically long timescale. Verified physically first (checked that a pure
   single-axis spin-up on each of the 3 axes shows axis 1 — the intermediate
   moment — growing perturbations by a factor of ~5×10⁴ over 20s while axes 0
   and 2 stay bounded, confirming the ODE implementation itself was correct)
   before changing the test's IC to an equal-in-*omega* perturbation, which
   generically avoids landing on the decaying eigendirection and flips within
   ~1s.

3. **Stage D/F convergence numbers in the test file were placeholders** copied
   from the LieSPHGP memory notes, which used a completely different inertia
   scale (I ~ 0.006–0.013 there vs. this project's I ~ 0.075–0.95). Recalibrated
   numerically against this project's actual inertia: Stage D needs Kp=1.0 over
   T=6s (not Kp=0.10/T=0.30s) to converge to <1% error here; Stage F needs
   K_R=K_p=1.0 over T=8s. Both now documented inline in `test_controllers.py` as
   `STAGE_D_KP`/`STAGE_D_T` constants with a comment explaining the mismatch, so
   a future reader doesn't mistake them for copy-paste of the PR's numbers.

`python export.py` runs end-to-end and produces real scenario JSON in `data/`
(free narrative: 800 frames, 6 segments, 8 background trajectories + 8
separatrices; controlled scenarios include `desired_H`/`desired_L`). Confirmed
the default racket geometry's inertia (`I ≈ [0.075, 0.875, 0.950]`) happens to
closely match the arbitrary `I_DEFAULT` used across the rigid_body/phase_portrait/
controllers test files — those were chosen independently (to loosely match the
whiteboard sketch's axis labels), so this is a coincidence worth knowing about,
not a hidden coupling between the test files and `racket_geometry.py`.

**Visual sanity check (throwaway, not committed):** before starting on the JS
renderer, rendered the actual exported JSON (not a separate hand-computed case)
with a matplotlib script in the same style as `dzhanibekov_display.py`, to catch
anything the numeric tests couldn't see (e.g. a quaternion sign/handedness bug
that passes every scalar invariant but looks wrong). Confirmed: the racket's
face color visibly flips red→green across "The flip" segment in the free
scenario, in lockstep with the yellow dot tracing the orange separatrix on the
sphere; and in the controlled-to-imax scenario, the racket visibly flips and the
dot converges exactly onto the target (green) fixed point by the end. No bugs
found — gives confidence the remaining work is a rendering-correctness problem
(quaternion convention, camera, styling), not a physics problem.

**Next step:** implement `app/src/core` (`DataLoader.js`, `Playback.js`) and
`app/src/scenes` (`RacketScene.js`, `SphereScene.js`) until `npm test` is green,
re-checking the `DataLoader.test.js` "should throw" cases flagged above along
the way. Still no visual/styling work.

## 2026-07-24 — App layer implementation, `npm test` green (21/21)

Implemented `DataLoader.js`, `Playback.js`, `RacketScene.js::quaternionToMatrix`,
`SphereScene.js::dotPosition` in that order, mirroring the physics-side
implementation order (data validation → playback state → rendering math).

- `DataLoader.validateScenario` mirrors `physics/export.py::scenario_to_json`'s
  checks (same required keys, same length-consistency logic) so both sides
  reject the same malformed shapes. Tightened the `DataLoader.test.js` "should
  throw" cases flagged as a caveat earlier today to assert on the actual error
  message (e.g. `/missing required keys.*R_cas/`), not just "throws something"
  — the earlier version would have passed against any stub. Added one more
  case (`accepts a valid controlled scenario with desired_H/desired_L
  present`) since the negative cases alone didn't prove the positive path for
  `mode: "controlled"` was reachable at all.
- `quaternionToMatrix` reindexes our scalar-first `[w,x,y,z]` convention into
  THREE.Quaternion's scalar-last `(x,y,z,w)` constructor, then lets THREE build
  the rotation matrix — passed all 6 cross-check cases against the Python
  `quat_to_R` fixture on the first try, including the two "hard" non-axis-
  aligned cases (120° about (1,1,1), an unnormalized input). No handedness or
  scalar-order bug.
- `Playback` implements `frameIndexAtTime` as a "last index ≤ t" binary search,
  and `tick()` returns `false` (and clamps, stops) once `currentTime` reaches
  the last frame — matches the free scenario's "plays once then stops" intent
  from `dzhanibekov_display.py` and works unchanged for a controlled scenario's
  frames too (nothing here depends on whether the scenario conserves energy).

**Still not done:** no wiring into `main.js`, no actual `THREE.Scene`/renderer/
camera, no visible output yet — `index.html` still just shows placeholder text.
Skipped a browser preview this round since there's nothing new to observe there
yet; the app becomes visually checkable once the scenes are wired into a real
render loop.

## 2026-07-24 — Render loop wired up, verified in-browser across all 4 scenarios

Before this could be wired up, found a real gap: the exported scenario JSON had
no static racket *mesh* (only per-frame quaternions), so Three.js had nothing to
build a shape from. Fixed by adding a `geometry` block to the schema
(`verts_body`/`handle_idx`/`hoop_idx`/`face_thickness`/`face_normal_body`) —
already fully resolved in the same body frame `frames.quaternion` rotates, so
the app never needs to know the racket's raw construction parameters or repeat
the inertia-tensor/eigenvector computation. Added
`racket_geometry.py::build_racket`'s `face_normal_body` (`evecs.T @ [0,0,1]`)
along with two regression tests proving it lands *exactly* on the imax axis
(not approximately) — guaranteed by the perpendicular-axis theorem since the
racket is planar. Propagated through `scenarios.py`, `export.py` (validated +
serialized like every other required key), `DataLoader.js` (mirrored
validation), and the JS sample fixture. Physics: 31/31 pytest. JS: 22/22 Vitest.

Built `RacketScene.createRacketMesh`/`updateRacketOrientation` and
`SphereScene.createSphereScene`/`updateDot`, and wired everything into
`main.js`: a scenario selector, play/pause, and two `THREE.WebGLRenderer`
panels sharing one `Playback` clock via `requestAnimationFrame`. Key
implementation choice: the racket's two offset faces are built *once* in
body-frame local space (`vertex ± face_thickness * face_normal_body`) rather
than recomputed in world space every frame like the matplotlib reference does —
valid because rotation commutes with a body-fixed offset (`R(v+tn) = Rv+tRn`),
and much cheaper (one `group.quaternion` assignment per frame instead of
rebuilding geometry).

Verified live in the Browser pane against all four generated scenarios:
- **Free narrative**: face color visibly flips red→green during "The flip",
  dot traces the separatrix in sync — matches the earlier matplotlib check.
- **Controlled→imax**: dot starts at the unstable (red) point, ends immediately
  adjacent to the target (green) point; racket ends in a fast clean spin.
- **Alignment** (ω\*=0, R\*=identity): the subtlest correctness check available
  — since the target angular momentum is exactly zero, the dot must converge
  not to a point *on* the sphere but to its exact *center*. It does.

One tool quirk, not an app bug: `computer.left_click` on the Play button
sometimes didn't register through the Browser pane despite correct coordinates
(confirmed via `getBoundingClientRect`); dispatching the click via
`element.click()` in `javascript_tool` worked every time. Noting in case it
recurs during later UI work.

Added a `racket-viz-app` entry to `LieSPHGP/.claude/launch.json` (alongside the
untouched old `tennis-racket-visualizer` entry) for `npm run dev` in
`racket_viz/app`; also set `vite.config.js`'s `publicDir: "../data"` so the
generated scenario JSON is served directly with no copying/duplication.

**Next step:** panels 3 (angular velocity vs. time) and 4 (current-vs-desired
control) are still unbuilt — currently a 2-panel app, not 4. Still no
layout/styling/design pass; that was always meant to come last, once the
simulation itself was verified correct, which it now is for panels 1 and 2.

## 2026-07-24 — Panel 3 (angular velocity vs. time)

Built `TimeSeriesPanel.js`: `computeOmega`/`computeOmegaDomain` as pure,
tested functions (3 new Vitest cases), plus an untested `drawTimeSeries` canvas
renderer — same split as `RacketScene`/`SphereScene` (test the math, verify the
drawing visually), since canvas/WebGL output isn't meaningfully unit-testable
in jsdom.

This panel is plain HTML5 `<canvas>` 2D, not Three.js — a 2D line chart doesn't
need a 3D scene, and no charting library was part of the approved architecture,
so it's ~100 lines of direct canvas drawing instead of a new dependency.

Color convention: ω[imin] green, ω[imid] red, ω[imax] royalblue — reusing the
sphere panel's existing stable(green)/unstable(red) fixed-point colors rather
than inventing a new palette, and matching the whiteboard sketch's ω1=green/
ω2=red/ω3=blue legend (ω2 = imid is always the unstable axis in this codebase's
convention, so "red" means the same thing in both panels).

Wired into `main.js` as a third canvas + panel; the omega series/domain are
computed once per scenario load (not per frame), and each frame draws the
three lines, a vertical time cursor, and a small marker per line at the current
value — mirroring the "yellow = current state" convention already used for the
sphere's moving dot. Verified live: cursor and markers track correctly in sync
with the racket flip and the sphere dot across the free-narrative scenario.

Physics: 31/31 pytest (unchanged). JS: 25/25 Vitest (+3).

**Next step:** panel 4 (current-vs-desired control) is the last panel —
requires deciding how to present "current H(t)/|M(t)| vs. desired_H/desired_L"
for controlled scenarios (and presumably nothing / a placeholder for free
scenarios, which have no target). After that: 4-panel grid layout, and only
then the design/styling pass explicitly deferred since the start.

## 2026-07-24 — Panel 4 (current vs. desired) — the 4th and final panel

User's answer to the "what does panel 4 show for free scenarios?" question:
unify around always plotting the current `H(t)`, varying only which reference
lines get drawn against it:
- **free**: 3 dashed lines at the natural equilibrium energies (`H_axis`),
  reusing panel 3's green(imin)/red(imid=separatrix)/blue(imax) colors — shows
  which energy band the conserved H(t) sits in.
- **controlled**: 1 dashed line at `desired_H` — a literal current-vs-desired
  convergence plot, directly answering the user's question about whether this
  should already show the IDA-PBC target (yes, via the already-exported
  `desired_H`).

This required adding `H_axis` (the 3 equilibrium energies) to free scenarios'
exported meta, mirroring `desired_H`/`desired_L`'s existing treatment for
controlled ones — validated in both `export.py` and `DataLoader.js` the same
way (required key, checked length), with matching pytest/Vitest coverage and a
new `test_export_rejects_free_scenario_missing_H_axis`.

Built `ControlPanel.js`: `computeH`/`referencesForScenario` as pure, tested
functions (4 new Vitest cases) + an untested `drawEnergyPanel` canvas renderer
(same test/verify split as every other scene module). Switched `#panels` from
a 1-row flex to a 2x2 CSS grid now that all 4 panels exist, arranged to match
the whiteboard sketch's layout (control top-left, racket top-right, sphere
bottom-left, angular velocity bottom-right).

Verified live for both modes:
- **Free narrative**: H(t) starts at `H_axis[imin]` (the first segment's
  energy), steps down for the remaining segments — all near `H_axis[imid]`/
  `H_axis[imax]`, correctly clustered near the bottom reference lines since
  only the imin-axis segment has much higher energy on this scale.
- **Controlled→imax**: H(t) dips first (redirecting the spin costs energy
  before the target axis picks up speed) then converges exactly onto the
  `desired_H` line, in sync with panel 3's ω-components crossing over and the
  sphere dot approaching the target fixed point. This is the clearest possible
  demonstration that all 4 panels are telling one physically consistent story.

Physics: 32/32 pytest (+1). JS: 30/30 Vitest (+5).

**All 4 panels now exist and are verified correct.** Nothing left on the
"physically accurate" side of the original brief. Remaining work is the
explicitly-deferred part: layout/visual design/styling (the greeting-card
aesthetic), and — if there's time — the "swipe to another system" idea from
the original request.

## 2026-07-28 — Interactive parameter sidebar + JS physics port

**Context.** User wants a real control surface (mass/geometry sliders,
start/end axis, control on/off, wind on/off, duration) with a "Run" button —
matching `summer-2026/rotate.py`'s interactive spirit plus the LieSPHGP
Stage D/E/F control/wind extensions. This breaks the "Python precomputes once,
JS replays a static file" assumption, since physics now has to run for values
nobody precomputed ahead of time.

**Decisions, discussed and confirmed before writing any code** (see the
approved plan for full reasoning):
- **Physics ported to JavaScript** — not Pyodide/WASM (keeps single-source
  Python but costs a large download), not a backend (zero-rewrite but
  reintroduces the server process this project deliberately avoided). The
  existing `physics/` Python package is kept as the frozen, already-tested
  oracle the JS port is cross-validated against — same fixture-based pattern
  already used for `quaternionToMatrix`, now extended.
- **Wind** = LieSPHGP Stage E's actual `disturbance_torque_std` model: one
  random body-frame torque sampled *once* per run and held constant (verified
  by reading `envs/tennis_racket_3d.py` on the `tennis-racket-effect` branch
  directly, not assumed) — not a per-step stochastic process. Confirmed to
  work in both free and controlled modes.
- **Start/end position** = axis pickers (imin/imid/imax), not free-form
  orientation.
- **Presets pre-fill sliders**, they don't fork the code path — a preset
  click and a manual slider tweak both end up calling the exact same
  `buildScenario(params)`.
- **"Run" produces one continuous trajectory**, not the old 6-segment
  bookended narrative. A real simplification (loses the per-phase captions),
  traded for genuine parameterizability.

**Implementation, module-for-module mirroring `physics/`** (eigen3x3 →
rigidBody → racketGeometry → phasePortrait → controllers → wind →
scenarioRunner, same dependency order as the original Python build): every
module's tests passed on the **first implementation attempt** — a very
different experience from the original Python build, and worth noting why:
the hard physics questions (the flip IC's growing-eigendirection subtlety, the
Stage D/F gain calibration, the numpy `assert_allclose` broadcast quirk) were
already found and fixed once, in Python, and this port could just reuse those
answers instead of rediscovering them. `eigen3x3.js` (no Python line to
translate — no JS equivalent of `numpy.linalg.eigh`) also passed immediately,
including its near-degenerate-eigenvalue test case.

Physics: 32/32 pytest (+1, `test_export_rejects_free_scenario_missing_H_axis`-style
coverage extended). JS: 74/74 Vitest (+41 new: eigen3x3, rigidBody,
racketGeometry, phasePortrait, controllers, wind, scenarioRunner).

**One real bug, found in the browser, not by any test**: `scenarioRunner.js`
returned `phasePortrait.buildStaticCurves()`'s camelCase output
(`stableFixedPoints`) unrenamed into the `background` field, but the schema
(and `SphereScene.js`) expect snake_case (`stable_fixed_points`). Every unit
test passed anyway because `DataLoader.validateScenario` never actually
checked `background`'s shape — only `meta`/`frames`/`geometry`. The failure
was silent (no console error visible via the automated browser tools; found
by manually chaining `buildScenario` → `validateScenario` →
`createSphereScene` step-by-step in the page console until one threw). Fixed
in two places: the actual key-renaming bug in `scenarioRunner.js`, and the
real gap in `DataLoader.validateScenario` (now validates `background`'s keys
and fixed-point shapes too) — plus a regression test for each, one asserting
the renamed keys directly and one asserting `validateScenario` now catches
this exact mistake if it recurs.

**Verified live** across combinations that had never been exercised before:
free+wind (confirmed no errors, first-ever run of this combination in either
language), a custom non-preset geometry/mass combination (still produces a
correct periodic Dzhanibekov flip), and the Control→imax preset (full `H(t)`
convergence onto `desired_H`, `ω` crossover, dot reaching the target fixed
point — all in sync, same as the earlier precomputed-scenario verification).

**Next step:** the app is now feature-complete for "physically accurate and
parameterizable." Only the explicitly-deferred design/styling pass (and the
optional "swipe to another system" stretch goal) remain.

## 2026-07-28 — First hands-on feedback round: 8 fixes, no styling yet

User tried the sidebar directly and reported 8 issues/requests, discussed one
at a time before any code changed (per their explicit request to "talk it out
first" on the ambiguous ones):

1. **Jagged/fuzzy Casimir sphere curves** (sphere wireframe itself was fine,
   explicitly keep that). Root cause: background trajectory/separatrix curves
   sit at exactly `|M|/R_cas == 1`, the same radius as the sphere mesh —
   classic WebGL z-fighting (coincident depth), not a data/integration
   problem. Fixed by rendering curves at `1.006x` the sphere's radius
   (`SphereScene.js`). Confirmed visually: curves are now clean thin
   anti-aliased lines; remaining visual density is legitimately from multiple
   overlapping sign-variant curves (same design as `dzhanibekov_display.py`),
   not an artifact.
2. **Dot "rides the separatrix down and up again."** Real bug, not a taste
   issue: `scenarioRunner.js`'s flip perturbation added a fixed `eps=0.02` in
   raw omega-space, symmetric on both stable axes — dimensionally disconnected
   from `R_cas` (order-of-magnitude too close to the exact separatrix for
   typical geometries) *and* structurally different from
   `dzhanibekov_display.py`'s actual construction (an M-space kick, dominant
   on ONE stable axis, tiny on the other — not symmetric). Rewrote to match
   that construction with `NEAR_SEP_PERTURB = 0.02` (fraction of `R_cas`) —
   between the too-close-to-tell-apart previous behavior and the reference
   script's `0.05`, per the user's "happy medium" ask.
3. **Control on, start==end axis → looks broken.** Kept the behavior (holding
   steady is correct), added an explicit caption ("Already at target — holding
   steady, nothing to correct") instead of hiding or blocking the choice.
4. **"Shouldn't H be an ellipsoid?"** — untangled a real conflation: the
   energy *level set* `H(M)=const` in 3D M-space is genuinely an ellipsoid,
   already implicit in panel 2 (the curves are its intersection with the
   Casimir sphere); panel 4's `H(t)` is a different, correctly-flat-for-free-
   motion scalar trace. Panel 4 unchanged. Added a new opt-in "Hamiltonian 3D"
   toggle in panel 2 instead (`SphereScene.createEnergyEllipsoid` /
   `updateEnergyEllipsoid`, semi-axes `sqrt(2*H*I_k)/R_cas`, rescaled per
   frame) — off by default since the user correctly predicted it'd get
   visually busy over time.
5. **Live slider preview + real line thickness**, both without a server:
   - `previewRacketGeometry()` in `main.js` rebuilds just `buildRacket()` and
     swaps the mesh on `input` events for handle/hoop/width/hoop-fraction
     sliders (not total mass — confirmed it cancels out of the COM math
     entirely and never touches vertex positions). No simulation involved.
   - Handle/hoop switched from `THREE.Line`/`LineLoop` to `TubeGeometry`
     meshes: `LineBasicMaterial.linewidth` is silently ignored by nearly
     every browser/GPU (long-standing WebGL spec quirk), so the old `2`
     genuinely did nothing. Purely a rendering change — physics still treats
     the racket as idealized 1D curves regardless of drawn thickness.

Also two smaller items folded in along the way:
- **A**: panels reflected across the vertical axis per the user's request —
  racket top-left, panel 4 top-right, angular velocity bottom-left, Casimir
  sphere bottom-right (previously the mirror image of this).
- **B**: consolidated "Control" from two places (preset button labels +
  a checkbox/dropdown pair lower down) into one real toggle-switch UI element
  (custom CSS, matching the existing Wind toggle's pattern exactly): Ending
  axis now hides/shows with the Control switch; Starting axis stays always
  visible (confirmed with the user — it's meaningful for free spin too, e.g.
  the Free flip preset has no controller at all but still needs a start axis).
- **Run now auto-plays** (previously left the user needing to separately hit
  Play after Run, which read as "nothing happened" — user flagged this
  directly): `renderDoc()` calls `playback.play()` immediately.

Physics: 32/32 pytest (unchanged — none of this touched `physics/`). JS:
74/74 Vitest (unchanged count; these were all rendering/UI/tuning changes, not
new pure-function surface, so no new unit tests were added for this batch —
verified each visually in the browser instead, including a debug-console
sanity check of the ellipsoid's actual scale values before confirming it
render correctly).

**Explicitly still deferred**: labels and general aesthetic polish — the user
reiterated they want the functional issues resolved first.

## 2026-07-29 — Presets/duration removed, Stop button, control readouts, rolling time windows

**Presets and Duration temporarily removed.** The user found the 4 preset
buttons unintuitive as a UI element (not obvious what each one *meant*) and
the Duration slider pointless once runs can just be stopped manually. Both
are commented out, not deleted, in `index.html` and `main.js` (with comments
explaining why), so they're easy to restore later. Runs now default to a
long fixed `DEFAULT_DURATION_SECONDS = 300`; a new **Stop** button
(pause + seek to t=0) sits below **Run**, distinct from **Pause** (freeze in
place). This was also motivated by wanting to exercise long integrations,
which is how the RK4 drift bug below (deferred, not yet fixed) was found.

**New: information back to the user for controlled runs.** Added to
`app/src/scenes/ControlMetrics.js`:
- `isAchievable(meta)` — an "achieved target" flag is only well-posed for
  controlled + no-wind + stable-target-axis scenarios (wind never lets
  corrective torque relax to zero; holding the unstable imid axis against a
  perturbation doesn't either). Everything else gets an ongoing effort
  readout only, never a "Done!" flag — confirmed explicitly with the user.
- `computeCumulativeWork` — running trapezoidal integral of controller
  torque · ω. Rectangle rule was tried first but failed a real cross-check
  test (`work(t) ≈ H(t)-H(0)` for a no-wind controlled run, since dH/dt =
  torque·ω exactly) by 0.063 vs. a required 0.05 precision — switched to
  trapezoidal, test passed.
- `computeAchievedIndex` — sustained-tolerance detection (must stay within
  5% of target for ≥1s, not just pass through momentarily).
- A small sidebar "Control function" panel (`drawTorquePanel`) plots torque
  magnitude vs. time with the same rolling window as the other time-series
  panels, plus a live "work done thus far" / "current torque" text readout
  (throttled to update every 500ms via `performance.now()`, not every
  animation frame) and the green "Done!" message when applicable.
- Also added: a live moments-of-inertia readout (`I1, I2, I3 = ...`) in the
  sidebar, updated on every geometry slider change.

**New: 10-second rolling time window for all 3 time-series panels.** The
angular-velocity and H(t) panels previously plotted the *entire* run (up to
300s) compressed into one fixed-width canvas, which the user described as
"compressed"/"overwhelmed with colors." Added `scenes/rollingWindow.js`
(`computeVisibleWindow`) — an oscilloscope-style window that grows from 0 to
10s then slides, with "now" always pinned at the right edge — and applied it
uniformly to `TimeSeriesPanel.js`, `ControlPanel.js`, and the new torque
panel.

**Two real bugs found during browser verification of the above (not present
in the physics — pure rendering/lifecycle bugs), both fixed:**
1. **Torque canvas never draws anything the first time Control is turned
   on.** `resize2DCanvas` sizes a canvas from its own `clientWidth`/
   `clientHeight`, but the torque canvas's container starts `display: none`
   (clientWidth/Height = 0) and was only ever resized once, at page load.
   Fixed by calling `resize2DCanvas(torqueCanvas, torqueCtx)` inside
   `refreshSubfieldVisibility()` whenever Control is switched on, so it's
   re-measured at the moment it actually becomes visible.
2. **Panel grid can permanently blow up to ~30,000px tall after a resize to
   a very small viewport.** Root cause: CSS grid items default to
   `min-height: auto`, which for a `<canvas>` child equals the canvas's own
   `width`/`height` *attributes* — and those attributes are themselves set
   from the panel's `clientHeight` (times devicePixelRatio) by
   `resize2DCanvas`/`renderer.setSize`. That's a feedback loop: one resize
   event with a transiently-small or off-by-a-factor clientHeight inflates
   the canvas attribute, which inflates the grid track's min-content size,
   which inflates the next resize's clientHeight, snowballing until the
   layout gets stuck huge — confirmed reproducible by resizing the browser
   to 309×383 and back to 1400×800. Fixed with `min-height: 0; min-width: 0;`
   on `.panel` in `index.html`, the standard fix for this class of
   flex/grid-vs-replaced-element bug. Re-verified the same shrink/grow
   sequence afterward: panel heights now correctly settle back to ~382px
   each instead of ~29,777px.

Physics: unchanged (32/32 pytest). JS: 92/92 Vitest (11 new tests for
`ControlMetrics.js`, 4 for `rollingWindow.js`; the two rendering bugs above
were caught by live browser verification, not unit tests, since they're
CSS-layout/canvas-lifecycle issues outside what Vitest's jsdom environment
exercises).

**Still deferred, not yet approved for implementation**: the long-integration
RK4 drift in `rigidBody.js`'s `integrateM` (fixed 10 substeps/frame; measured
~47% relative `|M|` drift by T=220s, matching `T_SEP_STATIC` — raising
substeps to 100 brings this down to ~0.0007% in a quick check). This is what
made the separatrix/trajectory curves look like "wiring" on long runs. The
user asked to diagnose and hold off implementing until given the go-ahead.

## 2026-07-29 (cont.) — Fixed the RK4 drift

Went with the second option floated above rather than just raising the fixed
substep count: `integrateFull` and `integrateM` no longer take "substeps per
output sample" as a bare fixed divisor. Both now default to
`Math.ceil(dtSample / MAX_STEP)` substeps, where `MAX_STEP = 0.001` is a cap
on the actual RK4 step size `h` — chosen because it's exactly the effective
`h` the main animated trajectory already used by default (T=300s, N=30000,
dtSample~0.01s, 10 substeps), which was already visually/numerically verified
good. This makes the fix a no-op for that path and only densifies the
under-resolved case: `integrateM`'s separatrix curves (T=220, N=900,
dtSample~0.245s) now get ~245 substeps instead of a fixed 10, and the
trajectory-family curves (T=80, N=900) get ~89 instead of 10.

An explicit `opts.substepsPerSample` override still works for both functions
(used by one new regression test to check the override path, and available
if a future perf issue calls for coarsening a specific call site
deliberately).

Added a regression test (`rigidBody.test.js`) reproducing the exact
production separatrix config (`T=220, N=900`, near-separatrix IC) and
asserting `|M|` stays within 1% of its initial value for the whole run — this
would have failed outright under the old fixed-10-substep code (~47% drift)
and passes now. Also re-verified visually in the browser: a long free-spin
run's separatrix curves are now clean and close on themselves instead of
visibly spiraling outward.

Physics (`physics/`, the frozen Python oracle) untouched — this was purely a
JS-port fixed-step-integrator bug, since the Python side already uses
`solve_ivp`'s adaptive RK45 and never had this problem. JS: 94/94 Vitest (2
new tests).

## 2026-07-30/31 — Geometry preview stability, rolling-window ramp-up, playback controls, analytic separatrix

**Head-to-mass ratio no longer moves the racket in the live preview.**
`hoopFraction` only changes mass distribution, not shape, but the preview
recentered on the mass-weighted center of mass, which shifts by up to a full
unit across the slider's range (confirmed: `com[0]` goes from -0.2 to 0.7 as
hoopFraction goes 0.2→0.8, with `evecs` staying exactly identity — a pure
translation, not a rotation). Fix: `buildRacket()` now also returns
`comUnweighted` (the plain average of vertex positions, ignoring mass —
depends only on shape, never on hoopFraction/totalMass), and
`previewRacketGeometry()` in `main.js` recenters on that instead. Verified in
Node that previewed vertex positions are now identical across hoopFraction
values to floating-point precision (1e-16). The real simulation (`Run`) is
untouched — it must still rotate about the true center of mass. Trade-off:
since preview and simulation now use different anchors, there's a one-time
jump the first time you touch a geometry slider after a run (switching from
"watching physics" to "editing shape" mode) — not further movement on
continued dragging, just that one switch.

**Rolling time windows no longer "compress" during ramp-up, and are now 5s.**
`computeVisibleWindow` used to map `[t0, currentTime]` onto the full panel
width while the run was younger than the window, so the effective
seconds-per-pixel scale shrank every frame until the window filled up —
visible as data compressing leftward. Fixed: the window is now always
exactly `windowSeconds` wide from the very first frame
(`windowEnd = max(t0+windowSeconds, currentT)`, `windowStart = windowEnd -
windowSeconds`), so the scale never changes. The "now" markers in
`TimeSeriesPanel.js`/`ControlPanel.js`/`ControlMetrics.js` no longer hardcode
to the panel's right edge either — they're drawn at `toX(t[currentIndex])`,
so they visibly slide in from the left and reach the edge exactly when the
window starts sliding. Default `windowSeconds` dropped from 10 to 5 per the
user's request (10 packed in too many oscillations to read comfortably).

**Playback controls consolidated into 3 identical sidebar buttons.** Removed
the top-left Play/Pause button (user found it hard to notice) and the
size/weight mismatch between Run and Stop (only `#run-button` had the bold
styling). Sidebar now has **Run** / **Pause** / **Reset**, all
`.sidebar-action` (same padding/weight/width), stacked where Run/Stop used to
be. Pause is the relocated old Play/Pause toggle (freezes in place, keeps all
graphs/readouts). Reset is exactly the old Stop behavior (pause + seek to
t=0), renamed since that's what it actually does.

**Separatrix curves replaced with the exact analytic solution (see prior
discussion in this session for the fuzziness diagnosis).**
`analyticSeparatrixCurves` in `phasePortrait.js` substitutes
`M[imid]=circleSign*R*tanh(lambda*t)`, `M[imin]=arcSign*A*sech(lambda*t)`,
`M[imax]=circleSign*arcSign*B*sech(lambda*t)` into Euler's equations and
solves for `lambda,A,B` in closed form — verified by direct substitution
(residual ~1e-11, floating-point noise) for arbitrary I/R_cas. No
integration, so no possible drift or the old ~9x pole-to-pole bouncing.

Two real bugs found and fixed during implementation, both worth recording:
1. **Wrong sign pairing.** Only `(circleSign, arcSign)` combinations where
   the sign relationship keeps `M[imin]` and `M[imax]` "correlated" the right
   way are genuine solutions of Euler's equations — of the 4 naive sign
   combinations tried first, half looked completely fine at first (conserved
   `|M|` and `H` exactly, since those are algebraic consequences of the
   ansatz regardless of sign) but had an Euler's-equation residual of ~2.8,
   i.e. were not real trajectories at all. **Conserving |M| and H is
   necessary but not sufficient evidence a curve is a genuine solution** —
   this is now called out explicitly in both the code comment and a
   regression test. The correct pairing was confirmed by direct numerical
   integration (`integrateM`) starting exactly on each candidate branch: the
   genuinely valid ones keep an exactly constant `M[imin]/M[imax]` ratio for
   the entire run, the invalid one doesn't correspond to any real trajectory
   at all.
2. **4 curves are actually needed, not 2.** The full sign-symmetry group has
   4 valid elements (a Klein four-group), not 2 — the true separatrix is 2
   great circles (`M[imin] = +r*M[imax]` and `M[imin] = -r*M[imax]`), each
   only half-traced by a single sign combination (one arc pole-to-pole
   through one quadrant); the other arc of the same circle needs the second
   valid combination. Missing this initially produced only 2 curves that
   were both (unknowingly) on the *same* circle.
3. **Rendered as thin tubes, not `THREE.Line`.** Even once correct, the
   curves were confirmed present (pixel-sampled directly off the canvas:
   thousands of correctly-colored pixels) but essentially invisible in
   practice — for geometries where the separatrix's excursion off the
   `[imid,imax]` plane is small relative to R_cas, a 1px line all but
   vanishes against the sphere's own wireframe grid. Switched to a thin
   `TubeGeometry` (`SEPARATRIX_TUBE_RADIUS = 0.01`), the same fix already
   used for the racket's handle/hoop and for the identical underlying reason
   (WebGL ignores `LineBasicMaterial.linewidth` on nearly every browser/GPU).

JS: 103/103 Vitest (7 new tests for `analyticSeparatrixCurves`, including one
that documents the invalid-sign-pairing bug directly, plus 1 new
`comUnweighted` regression test and updated `rollingWindow.js` tests).
Verified visually in the browser across two different geometries: 4 sharp
orange curves forming the classic "X" pattern through both unstable poles,
correctly reshaping with geometry changes.

## 2026-07-31 (cont.) — Starting perturbation for any axis, deduped blue curves, R_cas scale fix

**Blue trajectory curves deduped 8→4.** `familyCurves()` looped over both
`sa` and `sb` signs per stable axis, but `(sa, sb=+1)` and `(sa, sb=-1)`
trace the *same* closed orbit from two different starting phases (confirmed
numerically: one's IC lands within ~0.1% of R_cas of a point already on the
other's path). Drawing both wastefully integrated and rendered every loop
twice, and since each copy accumulates its own slightly-different RK4 error,
the near-duplicate pair didn't perfectly coincide -- rendering as a visibly
thicker/fuzzier line than a single clean pass, which is what the user was
noticing when comparing them unfavorably to the newly-sharp tube-rendered
separatrix. Fixed by fixing `sb=1` and only looping `sa`, giving exactly one
curve per actual fixed point (4 total: `+imin, -imin, +imax, -imax`).

**Starting perturbation now works from any axis, not just imid.** The old
kick was hardcoded to imin-dominant-plus-tiny-imax, applied only when
`startAxis===imid && !controlOn && !windOn`. Generalized into `kickedIC()`
(phasePortrait.js): dominant component stays on whichever axis is
`startAxis`, a user-controlled `kickFraction`-sized component goes on one of
the other two (still preferring imin, matching the exact original imid
tuning; falls back to imid when imin itself is the start axis), and a tiny
epsilon (now proportional to the kick, not a fixed constant) on the third.
Removed the old conditional entirely -- the kick is now always applied,
governed by a new **Starting perturbation** slider (0-0.1, step 0.002,
default 0.02) right after Starting axis. Wind is untouched, still layered on
independently on top.

Key design decision: the kick is scaled relative to `I[startAxis]*spinRate`
(that axis's own baseline angular momentum), **not R_cas**. R_cas is built
from I[imid] specifically; for imin/imax it can differ from that axis's own
baseline by an order of magnitude (verified: for this app's default I,
I[imin]*spinRate ≈ 0.47 vs R_cas ≈ 5.50). Scaling by R_cas uniformly would
have silently changed the baseline spin rate for every imin/imax scenario.
Scaling by the axis's own baseline instead means `kickFraction=0` exactly
reproduces today's plain `w0[axis]=spinRate` for any axis, and at imid's
default 0.02 it reproduces the original NEAR_SEP_PERTURB tuning almost
exactly (I[imid]*spinRate coincides with R_cas to ~5 significant figures).

**New live-updating "current orbit" background curve**, same blue as the
trajectory family, showing the actual orbit the current Starting axis/
perturbation choice will trace -- `currentAxisOrbitCurve()` integrates
`kickedIC`'s exact M0 for a short, adaptively-chosen duration derived from
the linearized rate `R_cas*sqrt(|(1/I[axis]-1/I[a])*(1/I[axis]-1/I[b])|)`
(imid: a real saddle growth rate, same formula `analyticSeparatrixCurves`
uses; imin/imax: a real linearized oscillation frequency, since those are
elliptic fixed points, not hyperbolic). imid's duration scales like
-ln(kickFraction)/rate (verified against actual integrated crossing times,
not just the bare formula, since small errors compound badly near a saddle);
imin/imax get a couple of real periods. Both are deliberately short --
verified nowhere near the point where a near-separatrix orbit would start
the multi-bounce fuzziness the exact separatrix curves had before being
replaced with a closed form. `buildBackgroundWithCurrentOrbit()` folds this
into `trajectories` (no schema changes) and is called both from
`buildScenario` (the real run) and a new `previewSphereBackground()` in
main.js (mirroring the existing live racket-shape preview), so the sphere
panel updates live as Starting axis / Starting perturbation / geometry
sliders change, without needing Run.

**Real bug found and fixed while verifying this**: the dot for a free
imin/imax start rendered deep inside the sphere instead of near its own
fixed point, because R_cas was unconditionally computed from imid
(`casimirRadius(I, imid, spinRate)`) regardless of which axis actually
started spinning. This was a pre-existing inconsistency (not introduced by
this work) that the new live background curves made obvious for the first
time, since they anchor prominently at the correct R_cas-scaled position
while the dot did not. Fixed for free runs: R_cas is now computed from the
actual `startAxis`.

**Extended to controlled runs too, same session.** Initially left controlled
scenarios on the imid-based convention (matching the user's earlier "don't
mess with it for now" on the related wind/P-controller steady-state-error
discussion), but on reflection this R_cas question is a genuinely separate,
purely-cosmetic issue -- confirmed `desired_H`/`desired_L` (what
`ControlMetrics.js`'s achieved-detection and panel 4's reference line
actually use) are computed directly from `omegaStar`/`I`, never from R_cas,
so changing R_cas's basis cannot affect whether control actually works,
Kp calibration, or the "Done!" logic -- only where things land in the sphere
panel. Since the controller's real target is `I[endAxis]*spinRate` (exactly
what `desired_L` already is), basing R_cas on `endAxis` for controlled runs
means a successful run's dot now lands exactly on the target's own
green/red marker instead of an unrelated imid-based radius -- and a
wind-driven steady-state offset now reads as what it actually is (a small,
visible miss right next to the target marker) instead of being swamped by a
scale mismatch an order of magnitude bigger than the real effect. Verified
in the browser: a no-wind imax->imin controlled run's dot now sits exactly
on imin's marker at convergence ("Done! Achieved at t=2.79s" unchanged); the
same run with wind on shows the dot close to but visibly offset from the
marker, with a nonzero "current torque" reading, matching the expected P-only
steady-state error exactly.

Physics unchanged. JS: 118/118 Vitest (13 new/updated tests: `kickedIC`,
`currentAxisOrbitCurve`, `buildBackgroundWithCurrentOrbit`, the trajectory
dedup, and the R_cas fixes for both free and controlled runs). Verified
visually in the browser: imid free-spin kick, imin/imax near-stable wobbles,
and controlled runs (with and without wind) all render correctly, with the
dot now landing at its actual fixed point instead of near the sphere's
center or an unrelated radius.

## 2026-07-31 (cont.) — Panel 4 becomes a real 3D Hamiltonian view, with a 2D/3D toggle

Panel 4 ("Current vs. desired (H)") was a flat H(t) line chart only; the
energy-ellipsoid visualization (semi-axes sqrt(2*H*I_k), the H=const level
set in M-space) lived as an optional overlay toggle ON the sphere panel
instead, added back when it was still an experiment (see the "Shouldn't H be
an ellipsoid?" entry). Promoted it into panel 4 as a proper second view:

- **New toggle** ("3D", top-right of panel 4) swaps between the existing 2D
  canvas and a new dedicated 3D one (`control-3d-canvas`), each with its own
  THREE.js scene/camera/renderer -- reusing the exact lesson from the torque
  panel's own display:none/block transition (a canvas sized from
  clientWidth/Height while hidden reads 0, so the toggle handler re-measures
  it the moment it becomes visible, not just on window resize).
- **Removed** the "Hamiltonian 3D" toggle and ellipsoid entirely from the
  sphere panel (`SphereScene.js`'s `createSphereScene` no longer builds or
  returns one) -- panel 4 is its dedicated home now, no redundant control.
- **New `createHamiltonianScene()` / `updateEnergyEllipsoid()`** (moved into
  `ControlPanel.js`, since panel 4 is now their only consumer): a faint
  reference sphere at R_cas (context -- the ellipsoid's own scale, normalized
  the same way, is directly comparable to it), a "current" ellipsoid that's
  ALWAYS visible and updates every frame (constant-sized exactly when H truly
  is conserved -- free, no wind -- and visibly resizing otherwise, since wind
  or control both do real work; this needed no mode-specific branching at
  all, since `updateEnergyEllipsoid` just reflects whatever H(t) actually
  is), a "desired" ellipsoid shown only for controlled scenarios (fixed at
  meta.desired_H, set once in `renderDoc` rather than every frame like
  `currentEllipsoid`, since a controlled target doesn't change mid-run), and
  a dot at the current M-position (reusing SphereScene.js's own
  `dotPosition`/`updateDot`, no duplication).

Verified in the browser: a free-spin run shows one white ellipsoid tracking
H(t) with the dot on its surface; a controlled run shows both ellipsoids
distinct early on (confirmed numerically too: H(0)=17.35 vs
desired_H=18.75 for a representative imid->imax run) and visually converging
into one orange-tinted shape once achieved (H(t) reaches desired_H exactly),
matching the "Done!" 2D-panel state. Toggling back to 2D restores the flat
chart correctly. No console errors across free/controlled/wind combinations.

JS: 121/121 Vitest (5 new tests: `createHamiltonianScene`'s default shape/
visibility, `updateEnergyEllipsoid`'s scaling and negative-H clamping).

## 2026-07-31 (cont.) — Auto-fit camera for panel 4's 3D view

User-reported: the energy ellipsoid could "blow way out of proportion" of
its panel, inconsistently. Reproduced directly (not guessed): with geometry
sliders pushed toward their minimums (all within normal ranges -- handle=1,
hoop length=1, hoop width=0.5, mass=0.3, head-to-mass=0.2) starting on imin,
`I=[0.0019, 0.0765, 0.0784]`, giving semi-axes `[0.99, 6.33, 6.41]` -- over
6x the reference sphere's radius. Root cause: for a free run, R_cas is now
based on the actual starting axis (see the earlier "R_cas scale fix" entry).
When that axis's own moment of inertia is disproportionately small relative
to the other two, R_cas becomes tiny, but the ellipsoid's other semi-axes
(governed by the larger moments) don't shrink to match -- confirmed H stays
exactly conserved throughout, so this is a real, correct consequence of the
normalization choice, not a data bug or rendering glitch.

Fix: `computeAutoFitDistance(maxExtent, fovDegrees, margin=1.3)` (new,
`ControlPanel.js`) computes how far back a perspective camera needs to sit
to frame content of a given bounding radius; `margin=1.3` was chosen so the
common maxExtent=1 case reproduces the original hardcoded distance (~3.17)
closely, so typical scenarios see no visible change. Every frame, `main.js`
takes the max of {reference sphere radius 1, both ellipsoids' max semi-axis,
the dot's distance from origin} and smoothly (10%/frame lerp, not a hard
snap -- wind can make the ellipsoid's scale jitter a bit frame to frame)
moves `controlCamera` along its original fixed viewing direction to the
right distance. Camera distance resets to the neutral default at the start
of every new Run, rather than carrying over a wildly different previous
run's distance.

Verification hit a real testing-harness snag worth recording: samples taken
a few seconds apart initially showed the camera distance frozen well short
of its target, looking like a bug in the smoothing logic. Turned out to be
Chrome throttling `requestAnimationFrame` for the *non-fronted* browser tab
being tested -- fronting the tab (`tabs_select`) immediately showed the
distance fully converged to the target, and the ellipsoid framed correctly
with no overflow. Good reminder for future verification: a frozen-looking
animated value during automated testing is worth checking against tab focus
before assuming the underlying logic is wrong.

JS: 124/124 Vitest (3 new tests for `computeAutoFitDistance`: reproduces the
original default distance, scales linearly with maxExtent, never returns
non-positive for a degenerate input).

## 2026-08-03 — Aesthetics pass (Observatory → Blueprint) and Cloudflare Pages deployment

Ran an aesthetics-mockup exploration (four static HTML color-theme previews
of the real panel layout, generated without touching any app code) so the
user could pick a direction before implementation. Landed on "Blueprint"
(deep navy, chalk-white ink, gold/cyan/coral accents) after trying
"Observatory" (dark neon) first and deciding against it. Implementation:
`app/src/theme.js` centralizes every canvas/WebGL color as one exported
object; `index.html`'s `:root` CSS variables mirror the same palette for
chrome (backgrounds, borders, sliders, buttons) -- the two are hand-kept in
sync (documented in both files) since there's no build-time bridge between
JS and CSS here. Swapping themes going forward means editing those two
blocks, not hunting hex literals across scene files.

Alongside the theme, fixed several rendering issues the user caught by
inspection: the WebGL panels (racket, sphere, 3D Hamiltonian) were defaulting
to THREE's black clear color instead of matching the 2D panels' own
`ctx.fillStyle`, so all three scenes now get `scene.background` set
explicitly to `THEME.panelBg`. The four grid panels could render at visibly
different heights depending on how many legend items each one wrapped to
(more legend items → more wrapped lines → less height left for the actual
`.panel` canvas box within the same fixed-height grid cell) -- fixed with a
fixed (not min-) height on `.panel-legend` plus an empty matching-height
spacer under the racket panel (which has no legend of its own). Removed the
"close to the tipping point" racket caption entirely per request (the
underlying `scenarioRunner.js` caption field is left alone -- unused
metadata now, same as the pre-existing `dot_color` field, not worth a
schema change for a UI-only removal). The energy panel's y-axis floored at
a slightly-negative padded minimum instead of exactly 0 even though H is
provably non-negative (a sum of squares over positive inertias), leaving a
dead strip of unreachable space below the "0" gridline -- now
`yMin = Math.max(0, yMin - pad)` so zero sits exactly at the plot's
bottom-left corner when the data justifies it. Both time-series panels
(`ControlPanel.js`, `TimeSeriesPanel.js`) also picked up small top/right
margins in `drawAxes`'s plot-area bounds -- the topmost y-tick and
rightmost x-tick labels were being drawn exactly at the canvas edge
(`plotTop=0`, `plotRight=width`), so half the label rendered off-canvas.

Separately, got `racket_viz` committed and pushed for the first time (it had
existed only locally until now) and set up Cloudflare Pages for a
shareable deployment link, hitting two unrelated blockers worth recording:

1. Cloudflare's build clones with submodules recursively, and this repo has
   had a dangling gitlink (`src/models/3D_SO3_Windy_Pendulum/ph_gp_ode_v2/JaxJD`,
   commit-mode tree entry with no matching `.gitmodules` entry) since its
   very first commit, on every branch -- so the clone step failed outright
   with "error occurred while updating repository submodules" before
   Cloudflare ever got to checking whether `racket_viz/` existed. Fixed by
   `git rm --cached`-ing the dangling gitlink, scoped to `feature/racket-viz`
   only (not `main`), since it's pre-existing and unrelated to this feature.
2. Cloudflare's GitHub OAuth login (used to sign in) is a separate grant
   from the GitHub App *installation* that actually authorizes repo access
   -- for an org-owned repo (`CVC-Lab/LieSPHGP`), installing/approving that
   app requires an org owner, not just the repo's contributors. Confirmed
   via GitHub's own settings (`Installed GitHub Apps` empty on the personal
   account, org-level installations page 404s for a non-admin) rather than
   guessed. Resolved once an org admin approved the pending install request.

Build settings landed on: root directory `racket_viz/app`, build command
`npm run build`, deploy command `npx wrangler pages deploy dist` (Cloudflare's
newer unified Workers+Pages UI needs an explicit Wrangler deploy command even
for a plain static site -- leaving it blank errors with "need deploy
command"). Also added `app/wrangler.jsonc` (name + `assets.directory`) so
`wrangler deploy` (not the old Pages-only `wrangler pages deploy`) can find
the project and static output automatically -- and passed
`--autoconfig=false`, since `wrangler deploy`'s default framework
auto-detection tries to wire up `@cloudflare/vite-plugin` and hard-errors
on anything older than Vite 6, which this project doesn't need at all for
a plain static build.

One more gotcha worth recording: Cloudflare's "Retry build" replays the
exact build-settings snapshot from when that build entry was originally
queued, not the project's current live settings -- so updating and saving
Settings → Build has no effect on a build you retry after the fact. Only a
genuinely new build (fresh push, or a new manually-triggered deployment)
picks up updated settings.
