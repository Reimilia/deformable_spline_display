/**
 * Casimir-sphere phase portrait: fixed points, separatrix, and orbit families.
 * Ported from physics/phase_portrait.py, EXCEPT the separatrix curves
 * themselves -- see analyticSeparatrixCurves below for why those are computed
 * from a closed-form solution instead of by porting physics/phase_portrait.py's
 * numerical construction.
 */
import { integrateM } from "./rigidBody.js";

/** @param {number[]} I @param {number} imid @param {number} spinRate @returns {number} */
export function casimirRadius(I, imid, spinRate) {
  const M_ref = [0.01 * I[0], 0.01 * I[1], 0.01 * I[2]];
  M_ref[imid] = I[imid] * spinRate;
  return Math.sqrt(M_ref[0] ** 2 + M_ref[1] ** 2 + M_ref[2] ** 2);
}

/** @param {number[]} I @param {number} R_cas @returns {number[]} H_axis[k] */
export function energyAtAxes(I, R_cas) {
  return [R_cas ** 2 / (2 * I[0]), R_cas ** 2 / (2 * I[1]), R_cas ** 2 / (2 * I[2])];
}

/**
 * Builds a small M-space perturbation off any starting axis, generalizing
 * what used to be an imid-only construction (dominant component on the
 * start axis, a `kickFraction`-sized component on one other axis, and a tiny
 * component on the third -- not a naive symmetric nudge, which for imid
 * specifically was found to sometimes land almost exactly on the saddle's
 * slow-decaying eigendirection; see DECISIONS.md).
 *
 * The kick is scaled relative to `I[startAxisIdx] * spinRate` (that axis's
 * OWN baseline angular momentum at `spinRate`), not R_cas. R_cas is built
 * from I[imid] specifically (casimirRadius), so for imin/imax starts it can
 * differ from I[axis]*spinRate by an order of magnitude or more -- scaling
 * by R_cas there would have silently changed the baseline spin rate for
 * every imin/imax scenario. Scaling by the axis's own baseline instead means
 * `kickFraction=0` reproduces today's plain `w0[axis]=spinRate` exactly, for
 * any axis -- and for imid specifically, I[imid]*spinRate coincides with
 * R_cas to ~5 significant figures, so this also reproduces the original
 * NEAR_SEP_PERTURB=0.02 tuning almost exactly at that same default value.
 *
 * @param {number[]} I [I1,I2,I3]
 * @param {number} imin @param {number} imid @param {number} imax
 * @param {number} startAxisIdx one of imin/imid/imax
 * @param {number} spinRate
 * @param {number} kickFraction fraction of the start axis's own baseline
 *   angular momentum; 0 gives an exact on-axis M0 (no perturbation at all)
 * @returns {number[]} M0
 */
export function kickedIC(I, imin, imid, imax, startAxisIdx, spinRate, kickFraction) {
  const baseline = I[startAxisIdx] * spinRate;
  const kick = kickFraction * baseline;
  const epsTiny = kick * 0.05;

  const others = [imin, imid, imax].filter((ax) => ax !== startAxisIdx);
  const kickAxis = others.includes(imin) ? imin : others[0];
  const tinyAxis = others.find((ax) => ax !== kickAxis);

  const M0 = [0, 0, 0];
  M0[kickAxis] = kick;
  M0[tinyAxis] = epsTiny;
  M0[startAxisIdx] = Math.sqrt(Math.max(0, baseline ** 2 - kick ** 2 - epsTiny ** 2));
  return M0;
}

/**
 * @param {number} Hval @param {number} Mmid @param {number[]} I @param {number} R_cas
 * @param {number} imin @param {number} imax @param {number} imid
 * @returns {number[]|null}
 */
export function icFromH(Hval, Mmid, I, R_cas, imin, imax, imid) {
  const Ia = I[imin];
  const Ib = I[imax];
  const Iu = I[imid];

  const S = R_cas ** 2 - Mmid ** 2;
  const E = 2 * Hval - (Mmid ** 2) / Iu;
  const denom = 1 / Ib - 1 / Ia;
  if (Math.abs(denom) < 1e-15 || S < 0) return null;

  const Mbsq = (E - S / Ia) / denom;
  const Masq = S - Mbsq;
  if (Mbsq < -1e-9 || Masq < -1e-9) return null;

  const M = [0, 0, 0];
  M[imin] = Math.sqrt(Math.max(0, Masq));
  M[imax] = Math.sqrt(Math.max(0, Mbsq));
  M[imid] = Mmid;
  return M;
}

/**
 * Closed-form heteroclinic-orbit solution of Euler's torque-free equations,
 * for the exact separatrix connecting the two unstable (imid) fixed points --
 * used instead of numerically integrating a near-separatrix orbit (as
 * physics/phase_portrait.py's build_static_curves does, and as this file's
 * own buildStaticCurves used to for the `separatrices` output).
 *
 * That numerical approach has a real problem: a trajectory started this close
 * to the true separatrix only takes a few seconds to cross from one unstable
 * pole to the other, so integrating it for T_SEP_STATIC=220s (a duration
 * picked for an unrelated reason -- see rigidBody.test.js's drift regression)
 * makes it bounce back and forth between the two poles many times over. Every
 * one of those ~9 crossings gets drawn as part of the same line, and since
 * our fixed-step RK4 doesn't land each crossing in *exactly* the same spot
 * (unlike physics/rigid_body.py's scipy solve_ivp at rtol=atol=1e-10), the
 * overlapping near-but-not-quite-identical crossings render as a fuzzy band
 * instead of one crisp curve, however finely the step size is tuned -- finer
 * steps only delay the divergence between crossings, they don't remove it.
 *
 * The exact separatrix sidesteps this entirely: substituting the ansatz
 * M[imid](t) = circleSign*R*tanh(lambda*t), M[imin](t) = arcSign*A*sech(lambda*t),
 * M[imax](t) = circleSign*arcSign*B*sech(lambda*t) into Euler's equations and
 * solving for lambda, A, B analytically gives a solution that satisfies them
 * exactly (to floating-point precision) for any I/R_cas -- no integration, no
 * per-crossing drift, so no possible fuzziness.
 *
 * The true separatrix (the H=H_sep level set intersected with the Casimir
 * sphere) is exactly 2 great circles, M[imin] = +r*M[imax] and
 * M[imin] = -r*M[imax] for a constant r derived from I (this drops out
 * directly from substituting |M|=R_cas into H=H_sep). Each circle is only
 * traced by ONE of the two (circleSign, arcSign) sign choices that share its
 * ratio -- (circleSign=+1: arcSign=+/-1) gives the two arcs of the +r circle,
 * (circleSign=-1: arcSign=+/-1) the two arcs of the -r circle -- confirmed
 * directly by substitution (the other 4 of the 8 possible sign combinations
 * are NOT valid solutions, despite sitting on the correct Casimir/energy
 * level, since |M|/H conservation is necessary but not sufficient). All 4
 * valid combinations are needed: each one only completes half of its circle
 * (from one pole to the other through a single quadrant), so 4 curves total,
 * matching the classic "X" phase-portrait picture of 2 crossing loops.
 *
 * @param {number[]} I [I1,I2,I3]
 * @param {number} R_cas
 * @param {number} imin @param {number} imax @param {number} imid
 * @param {object} [opts]
 * @param {number} [opts.nPts=900] samples per curve
 * @param {number} [opts.saturateFactor=10] curves run over t in
 *   [-saturateFactor/lambda, +saturateFactor/lambda] -- large enough that
 *   tanh/sech have saturated to within 1-tanh(10)~9e-9 of the poles, so the
 *   curve visibly completes its approach and settles, without an arbitrarily
 *   long unnecessary tail sitting at the pole.
 * @returns {number[][][]} exactly 4 curves (2 great circles x 2 arcs each),
 *   or [] if imin/imid/imax aren't 3 genuinely distinct moments of inertia
 *   (no well-defined separatrix in that degenerate case)
 */
export function analyticSeparatrixCurves(I, R_cas, imin, imax, imid, opts = {}) {
  const { nPts = 900, saturateFactor = 10 } = opts;
  const I1 = I[imin];
  const I2 = I[imid];
  const I3 = I[imax];

  const lambdaSq = R_cas * R_cas * (1 / I1 - 1 / I2) * (1 / I2 - 1 / I3);
  if (!(lambdaSq > 0)) return [];
  const lambda = Math.sqrt(lambdaSq);

  const k = ((1 / I2 - 1 / I3) * R_cas) / lambda;
  const P = (R_cas * lambda * I1 * I3) / (I3 - I1);
  if (!(k > 0) || !(P > 0)) return [];
  const B = Math.sqrt(P / k);
  const A = k * B;

  const Thalf = saturateFactor / lambda;

  function buildCurve(circleSign, arcSign) {
    const pts = new Array(nPts);
    for (let i = 0; i < nPts; i++) {
      const t = -Thalf + (2 * Thalf * i) / (nPts - 1);
      const sech = 1 / Math.cosh(lambda * t);
      const M = [0, 0, 0];
      M[imin] = arcSign * A * sech;
      M[imid] = circleSign * R_cas * Math.tanh(lambda * t);
      M[imax] = circleSign * arcSign * B * sech;
      pts[i] = M;
    }
    return pts;
  }

  const curves = [];
  for (const circleSign of [1, -1]) {
    for (const arcSign of [1, -1]) {
      curves.push(buildCurve(circleSign, arcSign));
    }
  }
  return curves;
}

/**
 * @param {number[]} I @param {number} R_cas
 * @param {number} imin @param {number} imax @param {number} imid
 * @param {object} [opts]
 * @returns {{ trajectories: number[][][], separatrices: number[][][],
 *   stableFixedPoints: number[][], unstableFixedPoints: number[][] }}
 */
export function buildStaticCurves(I, R_cas, imin, imax, imid, opts = {}) {
  const {
    nLevels = 0,
    epsFactor = 0.001,
    alphaSmall = 0.3,
    nStaticPts = 900,
    tTrajStatic = 80.0,
  } = opts;

  const eps = epsFactor * R_cas;
  const H_axis = energyAtAxes(I, R_cas);
  const H_sep = H_axis[imid];
  const stableAxes = [imin, imax];

  // sa picks which of the two fixed points (+axis or -axis) this loop
  // encircles; sb does NOT pick a different loop -- (sa, sb=+1) and
  // (sa, sb=-1) trace the SAME closed orbit, just starting from two
  // different phases along it (confirmed numerically: their trajectories
  // pass within ~0.1% of R_cas of each other). Looping over both wastefully
  // integrates and draws every loop twice -- each copy accumulates its own
  // slightly different RK4 error, so the redundant pair doesn't quite
  // coincide, rendering as a visibly thicker/fuzzier line than a single
  // clean pass. Fixing sb=1 keeps exactly one curve per actual fixed point.
  function familyCurves(Hk, T, N) {
    const curves = [];
    const icM = icFromH(Hk, eps, I, R_cas, imin, imax, imid);
    if (icM === null) return curves;
    const sb = 1;
    for (const sa of [1, -1]) {
      const M0 = [0, 0, 0];
      M0[imin] = sa * icM[imin];
      M0[imax] = sb * icM[imax];
      M0[imid] = icM[imid];
      curves.push(integrateM(M0, I, T, N));
    }
    return curves;
  }

  const trajectories = [];
  for (const ax of stableAxes) {
    const Hk = H_axis[ax] + alphaSmall * (H_sep - H_axis[ax]);
    trajectories.push(...familyCurves(Hk, tTrajStatic, nStaticPts));
  }
  for (const ax of stableAxes) {
    for (let k = 1; k <= nLevels; k++) {
      const alpha = (0.9 * k) / (nLevels + 1);
      const Hk = H_axis[ax] + alpha * (H_sep - H_axis[ax]);
      trajectories.push(...familyCurves(Hk, tTrajStatic, nStaticPts));
    }
  }

  const separatrices = analyticSeparatrixCurves(I, R_cas, imin, imax, imid, { nPts: nStaticPts });

  const stableFixedPoints = [];
  for (const ax of stableAxes) {
    for (const s of [1, -1]) {
      const p = [0, 0, 0];
      p[ax] = s * R_cas;
      stableFixedPoints.push(p);
    }
  }
  const unstableFixedPoints = [];
  for (const s of [1, -1]) {
    const p = [0, 0, 0];
    p[imid] = s * R_cas;
    unstableFixedPoints.push(p);
  }

  return { trajectories, separatrices, stableFixedPoints, unstableFixedPoints };
}

/**
 * Short, numerically-integrated curve showing the actual orbit a
 * `kickedIC`-perturbed start traces -- drawn in the same blue as the
 * trajectory family, so the sphere panel visually explains what a nonzero
 * startingPerturbation is about to do, for whichever axis is selected (not
 * just imid). Returns null when kickFraction is 0 (nothing to show -- an
 * exact on-axis start doesn't move).
 *
 * Duration is chosen from the linearized rate at the start axis,
 * `R_cas * sqrt(|(1/I[axis]-1/I[a])*(1/I[axis]-1/I[b])|)` (the same formula
 * analyticSeparatrixCurves uses for imid specifically -- there it's a real
 * saddle growth rate; for imin/imax it's a real linearized OSCILLATION
 * frequency instead, since those are elliptic, not hyperbolic, fixed points):
 *   - imid (saddle): transit time grows the closer the kick sits to the
 *     exact separatrix, roughly like -ln(kickFraction)/rate -- verified
 *     numerically against the actual integrated crossing time, not just the
 *     bare analytic formula, since near a saddle small formula errors can
 *     compound badly.
 *   - imin/imax (elliptic): a real, kick-independent period exists; a
 *     couple of periods shows the loop closes without wastefully overshooting.
 * Both cases are deliberately short integrations, well inside where a
 * near-separatrix orbit would start bouncing between poles repeatedly (that
 * bouncing, at a much smaller epsilon, is exactly what made the exact
 * separatrix curves fuzzy before they were replaced with a closed form).
 *
 * @param {number[]} I @param {number} R_cas
 * @param {number} imin @param {number} imid @param {number} imax
 * @param {number} startAxisIdx @param {number} spinRate @param {number} kickFraction
 * @param {object} [opts] @param {number} [opts.nPts=300]
 * @returns {number[][]|null}
 */
export function currentAxisOrbitCurve(I, R_cas, imin, imid, imax, startAxisIdx, spinRate, kickFraction, opts = {}) {
  const { nPts = 300 } = opts;
  if (!(kickFraction > 0)) return null;

  const others = [imin, imid, imax].filter((ax) => ax !== startAxisIdx);
  const rateSq =
    R_cas ** 2 * Math.abs((1 / I[startAxisIdx] - 1 / I[others[0]]) * (1 / I[startAxisIdx] - 1 / I[others[1]]));
  const rate = Math.sqrt(Math.max(rateSq, 1e-9));

  const T =
    startAxisIdx === imid
      ? Math.min(60, (3 / rate) * Math.log(Math.max(2, 1 / Math.max(kickFraction, 1e-4))) + 1 / rate)
      : ((2 * Math.PI) / rate) * 2;

  const M0 = kickedIC(I, imin, imid, imax, startAxisIdx, spinRate, kickFraction);
  return integrateM(M0, I, T, nPts);
}

/**
 * buildStaticCurves plus the live "current orbit" curve folded directly into
 * `trajectories` (same blue, no schema changes needed) -- the one function
 * both the real scenario build and the sidebar's live sphere-background
 * preview call, so what you see before hitting Run always matches exactly
 * what you're about to run.
 *
 * @param {number[]} I @param {number} R_cas
 * @param {number} imin @param {number} imax @param {number} imid
 * @param {number} startAxisIdx @param {number} spinRate @param {number} kickFraction
 * @param {object} [opts] passed through to buildStaticCurves
 * @returns {{ trajectories: number[][][], separatrices: number[][][],
 *   stableFixedPoints: number[][], unstableFixedPoints: number[][] }}
 */
export function buildBackgroundWithCurrentOrbit(
  I,
  R_cas,
  imin,
  imax,
  imid,
  startAxisIdx,
  spinRate,
  kickFraction,
  opts = {}
) {
  const background = buildStaticCurves(I, R_cas, imin, imax, imid, opts);
  const currentCurve = currentAxisOrbitCurve(I, R_cas, imin, imid, imax, startAxisIdx, spinRate, kickFraction);
  if (currentCurve) {
    background.trajectories = [...background.trajectories, currentCurve];
  }
  return background;
}
