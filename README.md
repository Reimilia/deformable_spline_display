# Deformation Lab

Interactive port-Hamiltonian checkpoint explorer. All checkpoint inference and
physics run in a background browser worker using **Pyodide WebAssembly, NumPy
float64, and Autograd**. GitHub Pages serves the complete application: a Python
server is not required.

The 58 model checkpoints from the supplied multilink experiments are selectable:
18 articulated multilink snapshots and 40 spline snapshots. The UI calls saved
measurement variants **FIM**, including all plot legends. Internal checkpoint IDs
keep their original names so provenance remains traceable.

## Run locally

Requires Node 24 and Python 3.12+ for the build helper. The browser runtime and
NumPy packages are downloaded once during the first build and are self-hosted.

```bash
cd web
npm ci
npm run build
npm run preview
```

The archive also includes a root-path production build. To open it immediately:

```bash
python3 -m http.server 8000 --directory web/dist
```

Then visit `http://localhost:8000`.

For development, run `npm run prepare:wasm` once, then `npm run dev`. The exported
parameters and task collections are included in `web/public/models/`. Open the
HTTP URL printed by Vite; opening `index.html` directly as a file will not work.

## Deploy

Follow [GITHUB_PAGES.md](GITHUB_PAGES.md). The included workflow exports the real
checkpoint parameters, validates the mathematical port against PyTorch and
inside WASM, builds the app, smoke-tests the static deployment, and publishes
`web/dist/` to GitHub Pages. Both repository paths and custom domains work.

## Controls and computation

Select a dynamics family, checkpoint, saved task set and task. Change the target
pose, initial rotation, integration steps, horizon, learned stiffness/damping,
FIM tolerance, FIM damping, or spline observation noise. Press **Run dynamics**
to compute a new trajectory. **Stop computation** terminates the worker;
**Reload WASM** starts a fresh interpreter. Large step counts and horizons take
longer, especially on mobile devices. First load downloads roughly 16 MB of
runtime assets, plus the selected checkpoint and its saved task collection.

All four panels share one playback cursor: configuration, phase space,
Hamiltonian/dissipation, and target/settling errors. Comparison uses the same
source task and the same control overrides, while the second model retains its
trained seam tolerance and FIM setting. Export JSON includes the settings,
checkpoint ID and original checkpoint SHA-256.

### Numerical implementation

PyTorch is used during export and validation. The `.pt` files are converted into
lossless NumPy `.npz` parameters: all trained weights, learned physical gains,
and exact stored snapshot state are retained, without quantization or retraining.
The browser recomputes the learned reference network and numerical dynamics for
every run; it does not replay saved trajectories.

The evaluation port preserves the original articulated inertia, anisotropic
drag, SE(2) update, FIM damping, hard constraint projections and compliant seam
forces. The spline port preserves the learned knot spans, periodic tangent
projection and gradients of the finite-iteration Sinkhorn/seam potentials.
Spline FIM is the display label for the saved measurement-closure variant;
it does not add a new Fisher damping operator to the spline model.

Observation noise uses NumPy's seeded PCG64 generator. Identical seeds reproduce
identical browser results, but the sample stream differs from Torch's generator.
Float64 parity tolerances account for native versus WASM numerical libraries.
Extreme configurations can be unstable; nonfinite output raises an error rather
than drawing fabricated values.

## Project layout

| Path | Purpose |
| --- | --- |
| `web/src/` | Canvas frontend, worker client and worker protocol |
| `web/public/python/browser_engine.py` | Browser-compatible checkpoint and dynamics evaluation |
| `web/public/models/` | Exported weights, task collections and provenance manifest |
| `python/` | Export, runtime staging and validation tools; optional PyTorch API |
| `python/vendor/` | Original multilink reference implementation and all 74 `.pt` files (58 models, 16 task caches) |
| `validation/` | Numerical and browser validation reports |
| `.github/workflows/pages.yml` | WASM validation, build and Pages publication |

The previous racket page, Three.js dependency/scenes, racket physics and tests,
old pendulum files, preview trajectories and Cloudflare configuration have been
removed. The original multilink checkpoint bytes and reference modules remain
unchanged.

## Regenerate models and validate

```bash
python3 -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r python/requirements.txt
python3 python/export_browser.py
python3 -m pytest python/tests -q
python3 python/validate_browser.py
cd web
npm ci
npm run prepare:wasm
npm test
npm run test:wasm
VITE_BASE_PATH=/deformation-demo/ npm run build
npx playwright install chromium
npm run test:browser
```

`test:wasm` uses the fixtures generated by `validate_browser.py`, so run the
Python validation first. They are regression fixtures only and are excluded
from the deployed frontend. The browser smoke test serves the frontend from
`/deformation-demo/` by default; use `TEST_BASE_PATH=/` after a root-path build.

See [validation/README.md](validation/README.md) for results and
[THIRD_PARTY.md](THIRD_PARTY.md) for runtime notices.
