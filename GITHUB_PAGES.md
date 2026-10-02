# Publish the WASM demo on GitHub Pages

1. Commit the project contents to your GitHub repository. Include `.github/`,
   `python/vendor/` (all checkpoint and task-cache files), `web/package-lock.json`
   and `web/public/models/`. Put these directories at the repository root.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
3. Push to the default branch, or open **Actions → Build WASM dynamics and
   publish GitHub Pages → Run workflow**.
4. Open the URL shown in the workflow's **github-pages** deployment.

The workflow checks out the code, installs CPU PyTorch, exports lossless
checkpoint parameters and task collections, validates all 58 models against
the original engine, tests the same evaluations inside actual WebAssembly,
builds Vite, tests the static frontend in Chromium, and publishes the site.
Validation reports are available in the `wasm-validation` Actions artifact.
A failed validation prevents publication.

The published site contains Python WASM, NumPy, Autograd, the evaluation port,
exported weights and saved task data. It does not contain Torch, pickle files,
the training code, regression fixtures, or an API server. No external CDN or
backend is contacted by the published demo. No COOP/COEP configuration or
SharedArrayBuffer is needed; computation runs in one dedicated worker.

## Repository paths and domains

`actions/configure-pages` supplies `base_path` to Vite. Worker scripts, WASM,
Python packages, model files and task collections all use that path. The static
smoke test uses the same prefix before deployment. User/organization pages and
custom domains use `/`; project pages use `/<repository>/`.

For a local project-path build:

```bash
cd web
VITE_BASE_PATH=/your-repository/ npm run build
```

A custom domain can be configured in GitHub Pages settings. If your repository
uses a `CNAME` file, put it in `web/public/CNAME` so Vite includes it in `dist/`.

## Updating checkpoints

Replace or add compatible checkpoints under `python/vendor/outputs/`, following
the original experiment structure. The build reruns `python/export_browser.py`
on every publication. Exported parameters are compressed Float64 NumPy arrays;
loading a selected checkpoint verifies its SHA-256 before evaluation.

GitHub Pages is static hosting, but Pyodide's WebAssembly interpreter executes
the Python numerical port locally in each visitor's browser. The `.pt` files
are build inputs, not browser executable models. The browser uses their exact
exported weights and recomputes trajectories for the chosen configuration.
