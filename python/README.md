# Reference engine and build tools

`vendor/multilink_ph/` and `vendor/outputs/` retain the supplied multilink model
code and checkpoint bytes. `engine.py` adapts the original PyTorch rollouts to
the frontend JSON contract and supplies the reference for validation.

- `export_browser.py`: export exact checkpoint parameters, saved task data,
  configuration metadata and original checkpoint SHA-256 to `web/public/models/`.
- `prepare_runtime.py`: stage the pinned npm Pyodide runtime and checksum-verified
  NumPy/Autograd wheels for self-hosted publication. Requires only Python stdlib.
- `validate_browser.py`: compare the evaluation port against PyTorch for all
  checkpoints, changed configurations and model comparisons; generate WASM test
  fixtures and a numerical report under `validation/`.
- `tests/test_engine.py`: reference rollout and adapter checks.
- `server.py`: optional original PyTorch API, available via `python run_demo.py`
  after installing `requirements.txt` and building `web/`. The WASM frontend
  runs independently; this API is for reference clients and diagnostics.

Only locally bundled, trusted `.pt` files are loaded by these build tools.
Uploaded file paths are never accepted from the frontend. Torch is absent from
browser inference. See the root README for the runnable commands.
