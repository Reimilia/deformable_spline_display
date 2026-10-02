/**
 * "Wind": a persistent aerodynamic disturbance torque. Ported from the
 * `disturbance_torque_std` model in envs/tennis_racket_3d.py on LieSPHGP's
 * `tennis-racket-effect` branch (used by Stage E's wind-robustness scenario).
 *
 * Confirmed from that source: this is NOT a per-step stochastic process (that
 * would be the separate, disabled "[WIND]" Brownian-motion block in the same
 * file) -- it's a single 3-vector sampled ONCE from N(0, windStd^2) per run
 * and held constant for the whole trajectory ("Sampled once per episode; held
 * constant throughout. Represents slow aerodynamic asymmetry."). That's what
 * makes it simple to port: just one more additive term in a torque_fn,
 * computed once before integration starts, not resampled per step.
 */

function gaussian(rng) {
  // Box-Muller transform from two uniforms in (0,1).
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * @param {number} std standard deviation per component [N*m]; 0 means no wind
 * @param {() => number} [rng] uniform [0,1) source, injectable for
 *   deterministic tests; defaults to Math.random
 * @returns {number[]} a single 3-vector disturbance torque
 */
export function sampleDisturbanceTorque(std, rng = Math.random) {
  if (std <= 0) return [0, 0, 0];
  return [gaussian(rng) * std, gaussian(rng) * std, gaussian(rng) * std];
}
