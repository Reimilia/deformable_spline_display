# Validation — 2026-10-02

- **69 reference adapter tests** passed.
- **10 frontend playback/plot tests** passed after removing the racket tests.
- **73 numerical parity cases** passed against the original PyTorch
  implementation: all 58 checkpoints, 13 changed-control variants and two
  cross-variant comparisons. Maximum absolute error across trajectories and
  diagnostics: `1.829e-11`.
- **74 actual WebAssembly cases** passed using Pyodide 0.28.3 and
  its NumPy/Autograd wheels. These include the same parity cases plus seeded
  observation noise. Maximum absolute error: `1.955e-11`.
- **Static browser smoke tests** passed from both `/deformation-demo/` and `/`: 58 selectable
  models, enabled physics controls, changed target/damping, comparison overlay,
  FIM legends, hard OOD spline task, observation noise, export, mobile layout,
  cancellation and worker restart. No API, preview or external requests occurred.
- All **74 original `.pt` files and 33 original reference modules** are byte
  identical to the supplied multilink repository.

The comparisons check configuration, momentum, world/body geometry, pose,
learned references, targets, and every exposed metric with `atol=2e-9` and
`rtol=2e-8`. The reported maximum is absolute and includes physical quantities
with different units. Coverage is representative, not an exhaustive sweep of
all task/configuration combinations.

`parity-report.json`, `wasm-report.json`, `browser-report.json` and
`browser-root-path-report.json` contain the
machine-readable results. The screenshots record desktop and mobile QA.
Regression fixtures in `wasm-cases.json.gz` are used by the test runner only;
they are never copied into the public deployment. The runtime does not fetch
these reports or fixtures.

Run the commands in the root README to regenerate all results. The same checks
run before publication in `.github/workflows/pages.yml`.
