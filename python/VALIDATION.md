# Validation record

## GitHub Pages update — 2026-10-02

- Generated and validated **116 previews**: feasible and hard-test playback
  for each of the 58 supplied models. Verified that all allowed static stage
  comparisons share the same saved target and initial state.
- Frontend: **153 tests passed**; includes repository-path asset resolution,
  hosted API URL handling and static-comparison rules.
- Production static build: passed with base `/deformation-demo/` and static
  mode enabled. Workflow YAML parsed; required preview and dependency files
  were checked against the repository's Git ignore rules.
- Browser: saved playback loaded under the repository prefix without any API
  calls. Verified feasible/hard selectors, both families, comparisons, export,
  mobile layout, and the original racket page.
- Browser and HTTP: connected the static frontend to a separate backend origin,
  accepted the configured CORS origin and JSON preflight, rejected an unlisted
  browser origin, recomputed changed damping, and returned to static previews.
  No uncaught page errors were recorded.
- These checks run locally. The GitHub deployment job runs after the repository
  owner enables Pages with GitHub Actions and pushes or dispatches the workflow.

Validated on 2026-10-01 with CPU PyTorch and Node 24.

- Python backend: **69 tests passed**. Every one of the 58 model checkpoints
  integrated through a full command interval using its saved integration
  resolution. This included pretrain/stage snapshots and hard-transfer models.
- Frontend: **147 tests passed**, including the original 144 racket tests and
  three new phase/time-alignment/domain tests.
- Vite production build: successful; both deformation and racket pages are
  included in `app/dist`.
- Live browser checks: desktop at 1440 px and mobile at 390 px; no uncaught
  page errors or mobile horizontal overflow. Verified default rollout,
  FIM legends, two-checkpoint overlay, shared initial state, target offsets,
  spline observation noise, coordinate/frame selection, scrubbing, JSON
  export, and an invalid-checkpoint API request returning HTTP 400.
- Source integrity: the 74 bundled `.pt` files and 33 vendor Python modules
  have the same SHA-256 hashes as their uploaded source files.

`qa/` contains screenshots, the browser check result and an exported example
comparison. Passing integration checks establishes that the stored models are
loaded and rendered consistently. It does not assert that every model solves
its task or converges under arbitrary parameter changes; the diagnostic panels
show its actual trajectory and errors.
