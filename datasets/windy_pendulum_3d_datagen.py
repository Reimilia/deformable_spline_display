import numpy as np
import pickle
import os
import argparse

import gymnasium as gym

import os
import importlib.util
import sys

from envs.windy_pendulum_3d import windy_pendulum_3d

# --- Import Environment ---
# try:
#     # If you keep it in an envs/ folder, update this path accordingly.
#     from windy_pendulum_3d import windy_pendulum_3d
# except ImportError:
#     # Fallback for the common project structure used in the original file:
#     try:
#         from envs.windy_pendulum_3d import windy_pendulum_3d
#     except ImportError as e:
#         raise ImportError(
#             "Could not import windy_pendulum_3d. "
#             "Place windy_pendulum_3d.py on your PYTHONPATH (or inside envs/)."
#         ) from e


# ─────────────────── Helper Functions ───────────────────

def to_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved data to {path}")


def from_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded data from {path}")
    return data


def arrange_data(x, t, num_points=2):
    """Arrange data to feed into neural ODE in small chunks.

    x : (num_us, T, N, D)
    t : (T,)

    Returns:
        x_stack : list of (num_us, num_points, N_windows, D) arrays (one per force)
                  OR a single stacked array
        t_eval  : (num_points,)
    """
    assert num_points >= 2 and num_points <= len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points - 1:
            x_stack.append(x[:, i:-num_points + i + 1, :, :])
        else:
            x_stack.append(x[:, i:, :, :])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack,
                         (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval


def _project_to_so3(R):
    """Numpy polar decomposition projection to SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1.0
        Rproj = U @ Vt
    return Rproj


def add_proper_noise_3d(clean_data, obs_noise_std, rng):
    """
    Apply geometrically-correct observation noise to 3D SO(3) pendulum data.

    For rotation matrices:  R_noisy = R @ expm(hat(eps)),  eps ~ N(0, sigma)
    For angular velocities: omega_noisy = omega + noise

    clean_data: (..., D)  where D = 9 (R) + 3 (omega) + 3 (u)
    """
    noisy_data = np.copy(clean_data)
    base_shape = clean_data.shape[:-1]
    D = clean_data.shape[-1]

    R_flat = clean_data[..., :9]
    omega = clean_data[..., 9:12]
    action = clean_data[..., 12:]  # u (3,)

    # --- Rotation noise via exponential map ---
    # Sample small rotation vectors eps ~ N(0, sigma^2 I)
    eps = rng.normal(0.0, obs_noise_std, size=base_shape + (3,))

    # Reshape for vectorized computation
    orig_shape = R_flat.shape
    R_flat_2d = R_flat.reshape(-1, 9)
    eps_2d = eps.reshape(-1, 3)

    R_noisy_flat = np.zeros_like(R_flat_2d)
    for i in range(R_flat_2d.shape[0]):
        R = R_flat_2d[i].reshape(3, 3)
        e = eps_2d[i]
        # hat map
        ex = np.array([[0, -e[2], e[1]],
                        [e[2], 0, -e[0]],
                        [-e[1], e[0], 0]], dtype=np.float64)
        # Matrix exponential for small rotation (Rodrigues)
        theta = np.linalg.norm(e)
        if theta < 1e-10:
            R_perturb = np.eye(3) + ex
        else:
            R_perturb = (np.eye(3)
                         + (np.sin(theta) / theta) * ex
                         + ((1 - np.cos(theta)) / (theta ** 2)) * (ex @ ex))
        R_n = R @ R_perturb
        R_noisy_flat[i] = R_n.flatten()

    noisy_data[..., :9] = R_noisy_flat.reshape(orig_shape)

    # --- Angular velocity noise ---
    omega_noise = rng.normal(0.0, obs_noise_std, size=base_shape + (3,))
    noisy_data[..., 9:12] = omega + omega_noise

    # Action is NOT corrupted
    noisy_data[..., 12:] = action

    return noisy_data


# ─────────────────── Sampling ───────────────────

def sample_windy_pendulum_3d(
    seed=0,
    timesteps=75,
    trials=50,
    u=(0.0, 0.0, 0.0),
    ori_rep="rotmat",
    friction_coeff=0.1,
    external_force_type="sine",
    external_force_std=1.0,
    external_force_direction=(1.0, 0.0, 0.0),
    wind_force_std=0.0,
    g=9.81,
    random_u=False,
    random_u_scale=1.0,
    varying_friction=True,
    **kwargs
):
    """
    Returns:
        trajs: (timesteps, trials, obs_dim + action_dim)
        tspan: (timesteps,)

    random_u_scale: half-width of the uniform distribution used when
        random_u=True. u ~ U(-scale, scale) per timestep. Defaults to 1.0
        for backward compatibility.
    """
    env = windy_pendulum_3d(
        g=g,
        external_force_type=external_force_type,
        external_force_std=external_force_std,
        external_force_direction=external_force_direction,
        friction_coeff=friction_coeff,
        varying_friction=varying_friction,
        wind_force_std=wind_force_std,
        ori_rep=ori_rep,
        render_mode=None,
        **kwargs
    )

    obs_dim = env.observation_space.shape[0]  # 12 for rotmat
    act_dim = env.action_space.shape[0]       # 3
    dt = env.dt
    timesteps = int(timesteps)
    trials = int(trials)

    trajs = []
    main_seed = int(seed)

    for trial in range(trials):
        valid = False
        retry_count = 0

        while not valid:
            if retry_count > 50:
                raise RuntimeError("Too many retries while generating a valid trajectory (solver instability or constraints).")

            # Reset with a new seed each retry
            obs, info = env.reset(seed=main_seed)

            # Choose initial control
            action_rng = np.random.default_rng(main_seed + 12345)
            if random_u:
                curr_u = action_rng.uniform(-random_u_scale, random_u_scale, size=act_dim)
            else:
                curr_u = np.array(u, dtype=np.float64).reshape(act_dim)

            traj = []
            # Record initial state
            x_init = np.concatenate((obs, curr_u.astype(np.float32)))
            traj.append(x_init)

            for t in range(timesteps - 1):
                obs, reward, terminated, truncated, info = env.step(curr_u)

                # Next action
                if random_u:
                    next_u = action_rng.uniform(-random_u_scale, random_u_scale, size=act_dim)
                else:
                    next_u = np.array(u, dtype=np.float64).reshape(act_dim)

                x = np.concatenate((obs, curr_u.astype(np.float32)))
                traj.append(x)

                curr_u = next_u

                if terminated or truncated:
                    break

            traj = np.stack(traj, axis=0)  # (timesteps, obs_dim + act_dim)

            # Validity checks
            if np.isnan(traj).any():
                retry_count += 1
                main_seed += 10
                continue

            # omega is the last 3 entries of obs for rotmat; enforce speed not saturated
            # NOTE: env.max_speed is no longer enforced by clipping in the env,
            # but is still kept as an attribute and used here as a soft validity
            # filter — trajectories whose |omega| exceeds max_speed are rejected
            # and resampled. With wind_force_std > 0 and no friction, retries
            # may happen frequently since omega can drift unbounded.
            omega = traj[:, 9:12]
            if (np.max(omega) < env.max_speed - 1e-3) and (np.min(omega) > -env.max_speed + 1e-3):
                valid = True
            else:
                print("hit speed limits, retrying with new seed...")
                retry_count += 1
                main_seed += 10

        trajs.append(traj)
        main_seed += 1

    env.close()

    trajs = np.stack(trajs, axis=0)          # (trials, timesteps, obs_dim+act_dim)
    trajs = np.transpose(trajs, (1, 0, 2))   # (timesteps, trials, obs_dim+act_dim)
    tspan = np.arange(timesteps) * dt

    return trajs, tspan


# ─────────────────── Dataset ───────────────────

def get_dataset(
    seed=0,
    samples=50,
    test_split=0.5,
    save_dir=None,
    us=((0.0, 0.0, 0.0),),
    ori_rep="rotmat",
    friction_coeff=0.1,
    external_force_type="sine",
    external_force_std=1.0,
    external_force_direction=(1.0, 0.0, 0.0),
    g=9.81,
    obs_noise_std=0.0,
    wind_force_std=0.0,
    timesteps=75,
    random_u=False,
    random_u_scale=1.0,
    varying_friction=True,
    **kwargs
):
    """
    Saves/loads a pickle with:
        data = {
            "x":           (num_us, T, N_train, obs+act),   # train (possibly noisy)
            "test_x":      (num_us, T, N_test,  obs+act),   # test clean
            "test_x_noisy":(num_us, T, N_test,  obs+act),   # test noisy
            "t":           (T,),
            "settings":    {...}
        }
    """
    if save_dir is None:
        raise ValueError("save_dir must be specified.")
    os.makedirs(save_dir, exist_ok=True)

    extforce_str = f"extforce-{external_force_type}-std{str(external_force_std).replace('.', 'p')}"
    obs_str = f"obs_noise{str(obs_noise_std).replace('.', 'p')}"
    wind_str = f"wind_force{str(wind_force_std).replace('.', 'p')}"
    fric_str = f"fric{str(friction_coeff).replace('.', 'p')}"
    var_fric_str = f"var_fric{str(varying_friction)}"
    if random_u:
        rand_u_str = f"random_u{str(random_u)}_uScale{str(random_u_scale).replace('.', 'p')}"
    else:
        rand_u_str = f"random_u{str(random_u)}"
    filename = f"wp3d_dataset_{extforce_str}_{fric_str}_{var_fric_str}_{obs_str}_{wind_str}_{rand_u_str}_steps{timesteps}.pkl"
    out_path = os.path.join(save_dir, filename)

    try:
        data = from_pickle(out_path)
        return data, out_path
    except FileNotFoundError:
        print(f"Building dataset at {out_path}...")

    trajs_force = []
    for i, u in enumerate(us):
        current_batch_seed = int(seed) + (i * 10000)
        trajs, tspan = sample_windy_pendulum_3d(
            seed=current_batch_seed,
            timesteps=timesteps,
            trials=samples,
            u=u,
            ori_rep=ori_rep,
            friction_coeff=friction_coeff,
            external_force_type=external_force_type,
            external_force_std=external_force_std,
            external_force_direction=external_force_direction,
            wind_force_std=wind_force_std,
            g=g,
            random_u=random_u,
            random_u_scale=random_u_scale,
            varying_friction=varying_friction,
            **kwargs
        )
        trajs_force.append(trajs)

    # (num_us, T, N, D)
    all_clean_x = np.stack(trajs_force, axis=0)

    # Train/Test split
    if test_split >= 0.5:
        split_ix = int(samples * 0.5)
    else:
        split_ix = int(samples * (1.0 - test_split))

    train_clean_x = all_clean_x[:, :, :split_ix, :]
    test_clean_x = all_clean_x[:, :, split_ix:, :]

    # Build data dict with keys matching 2D datagen format
    data = {}
    data['t'] = tspan
    data['settings'] = {
        'seed': seed,
        'samples': samples,
        'test_split': test_split,
        'us': us,
        'ori_rep': ori_rep,
        'friction_coeff': friction_coeff,
        'varying_friction': varying_friction,
        'external_force_type': external_force_type,
        'external_force_std': external_force_std,
        'external_force_direction': external_force_direction,
        'g': g,
        'obs_noise_std': obs_noise_std,
        'wind_force_std': wind_force_std,
        'timesteps': timesteps,
        'random_u': random_u,
        'random_u_scale': random_u_scale,
    }

    # Add observation noise
    if obs_noise_std > 0.0:
        print(f"Applying SO(3) observation noise (std={obs_noise_std})...")
        rng = np.random.default_rng(seed + 999)

        # TRAIN: Input is Noisy
        data['x'] = add_proper_noise_3d(train_clean_x, obs_noise_std, rng)

        # TEST: Input is Noisy, Target is Clean
        data['test_x'] = test_clean_x
        data['test_x_noisy'] = add_proper_noise_3d(test_clean_x, obs_noise_std,
                                                     np.random.default_rng(seed + 1999))
    else:
        print("No observation noise added.")
        data['x'] = train_clean_x
        data['test_x'] = test_clean_x
        data['test_x_noisy'] = test_clean_x

    to_pickle(data, out_path)
    return data, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 3D windy pendulum dataset on SO(3).")
    parser.add_argument("--save_dir", type=str,default="datasets/data/windy_pendulum_3d")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--test_split", type=float, default=0.5)
    parser.add_argument("--friction_coeff", type=float, default=0.5)
    parser.add_argument("--varying_friction", action="store_true")
    parser.add_argument("--external_force_type", type=str, default="sine", choices=["sine", "square", "random"])
    parser.add_argument("--external_force_std", type=float, default=0.0)
    parser.add_argument("--wind_force_std", type=float, default=0.1)
    parser.add_argument("--obs_noise_std", type=float, default=0.05)
    parser.add_argument("--random_u", action="store_true")
    args = parser.parse_args()

    data, path = get_dataset(
        seed=args.seed,
        samples=args.samples,
        timesteps=args.timesteps,
        test_split=args.test_split,
        save_dir=args.save_dir,
        friction_coeff=[0.5,0.5,0.5],
        varying_friction=args.varying_friction,
        external_force_type=args.external_force_type,
        external_force_std=args.external_force_std,
        wind_force_std=args.wind_force_std,
        obs_noise_std=args.obs_noise_std,
        random_u=args.random_u,
        us=((0.0, 0.0, 0.0),(1.0, 1.0, 1.0),(-1.0, -1.0, -1.0),),
    )
    print("Done.")
    print(f"Train (x):          {data['x'].shape}")
    print(f"Test (clean):       {data['test_x'].shape}")
    print(f"Test (noisy):       {data['test_x_noisy'].shape}")