# Third-party runtime notices

The numerical evaluation port derives from the supplied `multilink_ph` source.
The original reference implementation and checkpoints remain in `python/vendor/`.

The build redistributes these runtime components without source modification:

| Component | Pinned version | License / source |
| --- | --- | --- |
| Pyodide | 0.28.3 | MPL-2.0; https://github.com/pyodide/pyodide/tree/0.28.3 |
| Python | 3.13 (Pyodide build) | PSF; included in Pyodide's standard library |
| NumPy | 2.2.5 (WASM wheel) | BSD-3-Clause; wheel includes license notices |
| Autograd | 1.7.0 (WASM wheel) | MIT; wheel includes license notices |
| Future | Pyodide lockfile version | MIT; wheel includes license notices |

The matching runtime package hashes come from the pinned `pyodide-lock.json`.
The prepare script verifies every downloaded wheel against those hashes.
License notices from the wheels are retained in the redistributed packages.
Pyodide's license is included in `licenses/PYODIDE-MPL-2.0.txt`.

Development and reference tools include Vite, Vitest, Playwright and PyTorch.
Their licenses remain in their packages. They are not shipped as browser
runtime dependencies. Checkpoint distribution retains the provenance and
permissions of the supplied experiment repository.
