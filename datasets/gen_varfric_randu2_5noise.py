"""Generate the 5 datasets for the varying-friction + random-u(-2,2) study.

Specs (per request):
  varying_friction=True, friction_coeff=0.5
  external_force_std=0.0, wind_force_std=0.0
  random_u=True, random_u_scale=2.0   # u ~ U(-2, 2) per timestep
  obs_noise_std ∈ {0.01, 0.05, 0.1, 0.25, 0.5}
  samples=64 (matches both trainers' default), timesteps=20
"""
from __future__ import annotations
import os, sys

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_FILE_DIR)

from windy_pendulum_3d_datagen import get_dataset

NOISE_LEVELS = [0.01, 0.05, 0.1, 0.25, 0.5]
SAVE_DIR = os.path.join(PROJECT_ROOT, 'datasets/data/windy_pendulum_3d')

US = ((0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
      (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))

for noise in NOISE_LEVELS:
    print(f"\n=== building dataset obs_noise={noise} ===")
    data, path = get_dataset(
        seed=0,
        samples=64,
        timesteps=20,
        save_dir=SAVE_DIR,
        us=US,
        ori_rep="rotmat",
        friction_coeff=0.5,
        varying_friction=True,
        external_force_type="sine",
        external_force_std=0.0,
        wind_force_std=0.0,
        obs_noise_std=noise,
        random_u=True,
        random_u_scale=2.0,
    )
    print(f"  -> {path}")
    print(f"  -> x.shape={data['x'].shape}  test_x.shape={data['test_x'].shape}")
