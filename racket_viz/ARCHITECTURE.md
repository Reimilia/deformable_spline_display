# Architecture

## Physics lives in two places on purpose

`physics/` (Python) is the **frozen reference implementation**: racket
geometry → inertia tensor (`numpy.linalg.eigh`), integrated trajectories (free
or controller-driven, `scipy.integrate.solve_ivp`), and the static Casimir-
sphere phase portrait. It's no longer in the runtime path for the interactive
app, but it isn't dead code either — it's the already-tested oracle the JS
port below is cross-validated against (shared fixtures in
`physics/tests/fixtures/`, read by both `pytest` and `vitest`), and
`physics/export.py` can still produce standalone JSON if that's ever useful
again.

`app/src/physics/` (JavaScript) is a **from-scratch port** of that same
physics, run entirely client-side. This exists because the sidebar (see
below) needs to compute trajectories for parameter values nobody precomputed
— "Python precomputes once, JS only replays a static file" (the original
architecture, and still true of how `physics/` itself works) stopped being
possible once "Run" needed to accept arbitrary slider values. Confirmed with
the user: JS port, not Pyodide/WASM (would keep single-source-of-truth Python
but costs a large one-time download) and not a small backend (would keep
zero-rewrite but reintroduces a server process, undoing the static-hosting
simplicity this project deliberately chose after the prior attempt's
FastAPI+WebSocket+PyTorch approach). See `DECISIONS.md` for the full
reasoning and the one new piece of math with no Python line-by-line
counterpart (`eigen3x3.js`, since there's no JS equivalent of
`numpy.linalg.eigh`).

Both are structured identically on purpose — `app/src/physics/rigidBody.js`
mirrors `physics/rigid_body.py` function-for-function, same for
`racketGeometry.js`/`racket_geometry.py`, `phasePortrait.js`/`phase_portrait.py`,
`controllers.js`/`controllers.py` — so a bug fix or insight in one obviously
maps onto the other (e.g. the "equal-in-omega, not equal-in-M" flip initial
condition and the calibrated Stage D/F gains, both discovered in the Python
tests, carried over directly to the JS tests without rediscovery).

A single `Playback` cursor (`app/src/core/Playback.js`) drives all four panels
from one shared `t`, so they always show the same instant of the same
trajectory — there is one scenario active at a time, not four independent feeds.
This didn't change when the sidebar was added: `scenarioRunner.buildScenario()`
produces the exact same `{meta, frames, background, geometry}` shape a fetched
static JSON file used to (see Data contract below), so `main.js`'s rendering
code needed zero changes — only how that object is obtained changed (a local
JS function call instead of `fetch()`).

## Sidebar → scenarioRunner data flow

`index.html`'s sidebar collects: racket geometry/mass sliders, a head-to-mass
ratio, a starting axis (imin/imid/imax), control on/off (+ ending axis), wind
on/off (+ strength), and a duration. `main.js::readParamsFromSidebar()` reads
these into a plain params object and passes it to
`app/src/physics/scenarioRunner.js::buildScenario(params)`, which:

1. Builds the racket geometry (`racketGeometry.buildRacket`) for the chosen
   dimensions/mass — this is *not* cached across runs; a new geometry (and a
   fresh eigendecomposition) is computed every time, since any slider could
   have changed it.
2. Picks an initial `ω0` on the chosen starting axis. If that axis is the
   unstable imid one and nothing else would perturb it (no control, no wind),
   constructs a near-separatrix M-space kick instead of spinning exactly on
   the unstable equilibrium (which would just sit there forever): a dominant
   component on ONE stable axis plus a tiny one on the other
   (`NEAR_SEP_PERTURB` fraction of `R_cas`), matching
   `dzhanibekov_display.py`'s own near-separatrix segment construction — not
   a naive symmetric nudge, which either hugs the exact separatrix (looks like
   it "rides down and up" without ever cleanly flipping) or lands near the
   saddle's *decaying* eigendirection depending on the exact perturbation
   shape (see DECISIONS.md for both failure modes, found via user feedback and
   during the original Python flip-test debugging).
3. If control is on, builds a `PDBodyFrameController` targeting the chosen
   ending axis, with a gain from `controllers.defaultKpForAxisControl` —
   **not** a fixed constant (see DECISIONS.md: the calibrated Kp/K_R values
   from the Python tests were themselves specific to one inertia scale, so a
   user picking very different geometry/mass needs a gain that scales with
   their actual inertia to still converge).
4. If wind is on, samples one disturbance-torque vector
   (`wind.sampleDisturbanceTorque`, ported from LieSPHGP's Stage E
   `disturbance_torque_std` model — sampled **once**, held constant for the
   whole run, not a per-step stochastic process) and adds it to whatever
   torque is already being applied (zero, if control is also off).
5. Integrates one continuous trajectory (`rigidBody.integrateFull`, fixed-step
   RK4) and the static phase-portrait background
   (`phasePortrait.buildStaticCurves`), and assembles the result into the
   schema below.

Preset buttons (Free flip / Control→imin / Control→imax / Alignment) just
call `applyPreset()` with a fixed params object — including this project's own
already-calibrated `T` values (20s free flip, 6s Stage D axis-control, 8s
Stage F alignment) — then immediately run it, same as a manual Run click. This
was a deliberate simplification agreed with the user: presets pre-fill the
sidebar rather than being a separate code path, so there is exactly one way a
scenario ever gets built, whether from a preset or fully custom sliders.

Also confirmed with the user as a scope simplification: "Run" always produces
**one continuous trajectory**, not the fixed 6-segment bookended narrative the
original `dzhanibekov_display.py`-derived free scenario had (stable→wobble→
flip→separatrix→wobble→stable, each with its own caption). The "Free flip"
preset recreates the flip experience as a single run rather than that 6-part
story.

## Sidebar UI notes

- **Live geometry preview**: dragging handle length / hoop length / hoop
  width / head-to-mass ratio calls `main.js::previewRacketGeometry()`, which
  rebuilds only `racketGeometry.buildRacket()` and swaps the racket mesh —
  no trajectory (re-)simulation involved, so this is cheap enough to run on
  every slider `input` event with no server. Total mass is excluded from this
  live path: it cancels out of the center-of-mass math entirely and never
  changes vertex positions, only inertia magnitude.
- **Control toggle**: one real toggle switch (custom CSS, `.switch`), matching
  Wind's existing pattern. "Ending axis" shows/hides with it; "Starting axis"
  stays always visible regardless of Control, since it's meaningful for free
  spin too (e.g. the Free flip preset has no controller at all).
- **Panel grid order** does *not* match the panel numbering in the mapping
  table below: the grid is racket (top-left) → panel 4/current-vs-desired
  (top-right) → angular velocity (bottom-left) → Casimir sphere
  (bottom-right) — a deliberate left-right mirror of an earlier layout,
  requested by the user. `index.html`'s DOM order is the source of truth for
  grid position; `main.js` still refers to each canvas by id, so reordering
  the `.panel` divs doesn't touch any JS wiring.
- **"Hamiltonian 3D" toggle** (panel 2 only, off by default): overlays the
  energy level-set ellipsoid `{M : H(M) = const}` — semi-axes
  `sqrt(2·H(t)·I_k)/R_cas`, rescaled every frame via
  `SphereScene.updateEnergyEllipsoid` — so you can see directly that the
  `background` curves are exactly this ellipsoid's intersection with the
  Casimir sphere. Off by default because it gets visually busy fast.
- **Racket rendering**: handle/hoop are `THREE.TubeGeometry` meshes, not
  `THREE.Line`/`LineLoop` — WebGL silently ignores `LineBasicMaterial`'s
  `linewidth` on almost every browser/GPU, so a plain `Line` never renders
  thicker no matter what value is set. Purely a rendering choice; the physics
  still treats the handle/hoop as idealized 1D curves.
- **Sphere background curves** are drawn at `1.006x` the sphere mesh's radius
  (`SphereScene.js`), not exactly on it — they'd otherwise sit at the same
  radius as the sphere surface and z-fight against it (a classic WebGL
  coincident-depth artifact that looks like speckled/broken lines, not a
  problem with the underlying 900-point integration).

## Data contract

One JSON file per scenario, shape:

```jsonc
{
  "meta": {
    "mode": "free" | "controlled",
    "I": [I1, I2, I3],                 // principal moments of inertia
    "R_cas": 1.5,                       // |M| at t=0 (NOT necessarily conserved -- see below)
    "segments": [
      { "start": 0, "end": 130, "label": "...", "caption": "...", "dot_color": "#44ee66" }
    ],
    // present only when mode == "controlled":
    "target_axis": 2,                   // or "target_R": [[...]] for alignment
    "desired_H": 0.123,                  // panel 4 reference line: the controller's target
    "desired_L": 1.5,
    // present only when mode == "free":
    "H_axis": [H_imin, H_imid, H_imax]   // panel 4 reference lines: the 3 natural
                                          // equilibrium energies (H_axis[imid] == H_sep)
  },
  "frames": {
    "t": [0.0, ...],
    "quaternion": [[q0, q1, q2, q3], ...],   // scalar-first, body->world orientation
    "M_body": [[M1, M2, M3], ...]            // angular momentum in body frame
  },
  "background": {
    "trajectories": [[[M1,M2,M3], ...], ...],
    "separatrices": [[[M1,M2,M3], ...], ...],
    "stable_fixed_points": [[...], ...],
    "unstable_fixed_points": [[...], ...]
  },
  "geometry": {
    "verts_body": [[x,y,z], ...],        // static racket mesh, in the SAME body
                                           // frame frames.quaternion rotates
    "handle_idx": [0, 1, ...],            // indices into verts_body
    "hoop_idx": [40, 41, ...],
    "face_thickness": 0.04,
    "face_normal_body": [0.0, 0.0, 1.0]   // unit normal to the racket face, body frame
  }
}
```

`geometry` is static (one copy per scenario, not per-frame) and already fully
resolved in the principal-axis frame -- the app applies `quaternionToMatrix(q)`
to `verts_body` directly frame-by-frame; it does not need to know the racket's
raw construction parameters (handle length, hoop radius, etc.) or repeat the
inertia-tensor/eigenvector computation itself. `face_normal_body` looks like a
hardcoded `[0,0,1]` by coincidence for the current geometry: the racket is
planar (every vertex has z=0 before the principal-axis rotation), so by the
perpendicular axis theorem the out-of-plane axis is automatically a principal
axis with the largest moment (`imax`) -- see
`racket_geometry.py::build_racket`'s docstring and
`test_face_normal_is_unit_and_aligned_with_imax_axis`. It's still exported
explicitly (not assumed in JS) so a future non-planar geometry wouldn't
silently break rendering.

**Free scenarios**: `|M_body(t)|` and `H(t) = 0.5 · ω(t)·(I·ω(t))` are exactly
conserved (Casimir + energy). `R_cas` is the single background sphere radius for
the whole scenario, and the phase-portrait `background` curves live on it.

**Controlled scenarios**: an external body-frame torque is doing work on the
system, so `|M_body(t)|` and `H(t)` are *not* conserved — the yellow dot on the
sphere panel visibly leaves the `background` sphere it started on. `R_cas` in
`meta` is only the *initial* Casimir radius (for placing the static background
geometry, which is drawn once and does not itself move); `desired_H`/`desired_L`
are the controller's target values, plotted as reference lines panel 4 compares
the live `H(t)`/`|M(t)|` against. `rigid_body.integrate_full`'s `torque_fn` hook
is what makes this a real integrated trajectory rather than an interpolated one.

## Panel ↔ source mapping

| Panel | Renders | Reference implementation (Python, oracle) | JS port (runtime) | Reference |
|---|---|---|---|---|
| 1. Racket (3D) | `frames.quaternion` applied to the racket mesh | `racket_geometry.py`, `rigid_body.py` | `racketGeometry.js`, `rigidBody.js` | `dzhanibekov_display.py` §2-4, `dzhanibekov_8k.mp4` |
| 2. Casimir sphere | `background` curves/fixed points (static) + `frames.M_body` (moving dot) | `phase_portrait.py` | `phasePortrait.js` | `dzhanibekov_display.py` §5,8, `dzhanibekov_8k.mp4` |
| 3. Angular velocity vs. t | `frames.M_body / I` plotted over `frames.t` | (derived from `frames`, no new physics) | `TimeSeriesPanel.js::computeOmega` | `rotate.py`'s `ax_omega` panel |
| 4. Current vs. desired | `H(t)` (from `frames.M_body`/`meta.I`, always) vs. reference lines: `meta.H_axis` (free) or `meta.desired_H` (controlled) | `controllers.py` | `controllers.js`, `wind.js` (new, no Python equivalent yet) | LieSPHGP PR #1, `tennis-racket-effect` branch, Stage D/E/F |

Each panel also gets a caption strip (visual style inspired by
`dzhanibekov_display.py`'s boxed captions), with text authored per
`meta.segments[i].label`/`.caption` for this project.

## Controller provenance

`controllers.py` (and its JS port `controllers.js`) is adapted from
`src/models/3D_SO3_Tennis_Racket/ph_nn_ode_v2/controller_stageD.py` and
`controller_stageF.py` on LieSPHGP's open PR #1 (branch `tennis-racket-effect`,
commit `53fe83c`). Only the analytic-`g` path is kept (`g = I3`, i.e. direct
body-frame torque actuation) — the optional learned-`g_net`/PyTorch branch from
the original files is dropped, since neither the precompute pipeline nor the JS
port has a live model in the loop. "Three control functions" = the same law
validated at three target axes (e1/e2/e3, Stage D); "alignment" = the full
(R\*, ω\*) law (Stage F).

`wind.js` is adapted from the same branch's `envs/tennis_racket_3d.py`
(`disturbance_torque_std` / `self._disturbance_torque`, used by Stage E's
wind-robustness scenario) — no Python module in *this* repo has it yet, since
wind didn't exist as a feature until the sidebar needed it.

## eigen3x3.js — the one module with no Python line to translate

`racket_geometry.py` diagonalizes the inertia tensor with `numpy.linalg.eigh`
(LAPACK); there's no equivalent to import in the browser. `app/src/physics/
eigen3x3.js` implements the standard closed-form symmetric-3x3
eigendecomposition (characteristic cubic + trigonometric solution) instead —
exact for the 3x3 case, no iterative-convergence concerns. This got the most
dedicated test attention of any new module (see `eigen3x3.test.js`): known
diagonal matrices, a hand-built matrix with real off-diagonal terms verified
via the `A@v = λv` property (not exact eigenvector values, which aren't unique
up to sign), a near-degenerate case, and a cross-check fixture against
Python's actual `racket_geometry.build_racket()` eigenvalues. Note that
fixture is a weaker check than it looks: this racket's geometry is always
exactly diagonal already (planar + mirror-symmetric about the handle axis), so
every real matrix this project produces has *trivial* eigenvectors — the
hand-built rotated case is what actually exercises the general algorithm.
