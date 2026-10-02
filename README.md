# LieSPHGP

## Interactive deformation demo

**GitHub Pages deployment is included.** See [GITHUB_PAGES.md](GITHUB_PAGES.md).
The workflow publishes the frontend with 116 saved rollouts covering all
58 checkpoints. A hosted Python service can be connected for live parameter
changes; local use with `python run_demo.py` remains available.

This version includes **Deformation Lab**, using the supplied multilink and
spline checkpoints as selectable learned dynamics. The built frontend and all
58 model checkpoints are bundled. The original racket demo is available from
the top-right link.

From this repository's root, install the Python dependencies and start:

```bash
python -m pip install -r racket_viz/multilink_backend/requirements.txt
python run_demo.py
```

Open **http://127.0.0.1:8765**. Node is only needed when modifying the frontend.
The new demo uses CPU PyTorch, NumPy and SciPy; its runtime does not require
Pinocchio, JAX, Gymnasium, or a GPU. For a CPU-only PyTorch installation, install
PyTorch from its CPU wheel index before installing the requirements above.

- Choose **Articulated multilink** or **Deformable spline** and a checkpoint.
- Choose a feasible, hard OOD, training or validation task from the original
  saved task sets. Transfer checkpoints initially select hard OOD testing.
- Change target translation/rotation, initial rotation, horizon, integration
  resolution, learned damping/stiffness, and the applicable FIM controls.
- Press **Run dynamics** to recompute with the actual stored model weights.
- Inspect world/body configuration, coordinate–momentum phase trajectories,
  Hamiltonian components, dissipation, errors and settling with one shared time
  cursor. Pause, scrub, loop, change playback speed or export the rollout JSON.
- Select a second checkpoint to overlay it on the same saved task. Comparison
  is available within each dynamics family; the second model retains its
  trained closure tolerance and FIM setting.

Measurement-case labels and legends display **FIM**. See
[the implementation notes](racket_viz/multilink_backend/README.md) for the
checkpoint stages, numeric conventions and parameter meanings.

### Frontend development

Keep `python run_demo.py` running in one terminal. In another:

```bash
cd racket_viz/app
npm ci
npm run dev
```

Vite proxies `/api` to the Python service on port 8765. After editing, run
`npm run build` to refresh the bundled frontend served by `run_demo.py`.

### Tests

```bash
python -m pip install pytest
python -m pytest racket_viz/multilink_backend/tests -q
cd racket_viz/app
npm ci
npm test
```

The backend tests integrate every supplied model checkpoint, validate frame
alignment, compare the spline adapter with the original integrator, and check
overlay isolation and observation-noise reproducibility.

## Original LieSPHGP project

Stochastic Port-Hamiltonian Neural Networks for learning dynamics on Lie groups (SO(3), SE(3)).

## What's inside

- **envs/** — Gym environment for a 3D windy pendulum on SO(3).
- **datasets/** — Scripts to generate and plot trajectory data.
- **src/models/** — Models trained on the windy pendulum:
  - `ph_nn_ode_v2` — port-Hamiltonian neural ODE
  - `ph_gp_ode_v2` / `ph_gp_sde` — Gaussian-process variants
  - `neural_sde` — neural SDE baseline
- **src/utils/** — Shared helpers, including JAX implementations of GPs, neural nets, and Lie-group integrators.

## Quick start

1. Generate data:
   ```bash
   python datasets/windy_pendulum_3d_datagen.py
   ```
2. Train a model, e.g.:
   ```bash
   python src/models/3D_SO3_Windy_Pendulum/ph_nn_ode_v2/train.py
   ```
3. Compare models:
   ```bash
   python src/models/3D_SO3_Windy_Pendulum/ode_make_comparison_v2.py
   ```

## Requirements

Python 3.10+, NumPy, JAX, PyTorch, Gymnasium.
