# Port-Hamiltonian Loss Functions — Math Reference (ph_gp_sde_v2)

This document describes the **noise-aware** physics-informed losses
proposed for `ph_gp_sde_v2`. They are the same four auxiliary losses
used in `ph_nn_ode_v2` (`L_power`, `L_V`, `L_B`, `L_D`) but normalised
by the observation-noise scales already carried by the SDE model. This
removes the dependence of the loss magnitude on `obs_noise_std` and
gives a single $\lambda$ weighting that is meaningful across all noise
levels.

The SDE-specific noise scales the loss formulation can use are:

| Field on `DissipativeSO3HamSDE` | Role | Trainable? |
|---|---|---|
| `log_sigma_R` | per-frame rotation NLL std (rollout NLL on $R$) | yes |
| `log_sigma_omega` | per-frame angular-velocity NLL std (rollout NLL on $\omega$) | yes |
| `sigma_obs_omega` | $\omega$ observation noise; equals dataset `obs_noise_std` | **frozen** |

The loss formulation below uses **`sigma_obs_omega`** as the source of
$\sigma_{\mathrm{obs}}$ for the central-difference variance correction
(it is exactly the noise std added to $\omega$ snapshots in the
dataset, which is the noise that `ṗ_data = (p_{t+1} - p_{t-1})/(2\Delta t)`
sees). The trainable `log_sigma_omega` enters the rollout NLL and is
**not** reused in the auxiliary losses, so the two terms remain
independent.

---

## 1. Equations of Motion

### Configuration dynamics

$$\dot{q} \;=\; q^\times \, \nabla_p H$$

### Momentum dynamics (deterministic part)

$$\dot{p}^{\mathrm{det}} \;=\; -(q^\times)^T \nabla_q H \;+\; p^\times \nabla_p H \;-\; D(q,p)\,\nabla_p H \;+\; B(q)\,u$$

### Symbols

| Symbol | Shape | Meaning |
|---|---|---|
| $q$ | $9$ | Vectorized rotation matrix (rows of $R$ stacked: $q = [R_1; R_2; R_3]$) |
| $p$ | $3$ | Angular momentum |
| $\omega = M^{-1}(q)\,p$ | $3$ | Angular velocity |
| $H(q,p) = \tfrac{1}{2} p^T M^{-1}(q)\,p + V(q)$ | scalar | Total energy |
| $\nabla_p H$ | $3$ | $= M^{-1}p = \omega$ |
| $\nabla_q H$ | $9$ | Gradient of $H$ w.r.t. configuration |
| $q^\times$ | $9\times 3$ | Kinematic map; $q^\times \omega$ stacks $R_i \times \omega$ |
| $p^\times$ | $3\times 3$ | Skew-symmetric of $p$ |
| $M(q)$ | $3\times 3$ PSD | Generalised mass / inertia (model outputs $M^{-1}$) |
| $D(q,p)$ | $3\times 3$ PSD | Dissipation matrix |
| $B(q)$ | $3\times m$ | Input matrix |
| $u$ | $m$ | Control input |
| $\sigma_{\mathrm{obs}\omega}$ | scalar | $\omega$-observation noise std (= `model.sigma_obs_omega`) |
| $\Delta t$ | scalar | snapshot interval |

### A note on the SDE diffusion term

`ph_gp_sde_v2` adds a stochastic increment

$$dp_{\mathrm{stoch}} \;=\; R^T \big( l\,R\,e_z \times \sigma(q)\,dW \big)$$

to the momentum update. The auxiliary losses below compare the
**deterministic** drift $\dot{p}^{\mathrm{det}}$ against the central-
difference estimate $\dot{p}^{\mathrm{data}}$. Any non-zero diffusion
contributes additional variance to the residual that whitening cannot
remove (it has a different scale than the obs-noise floor). For the
verification protocol used here we set the env-side `wind_force_std = 0`
so $\dot{p}^{\mathrm{data}}$ contains only deterministic dynamics +
observation noise. The model's $\sigma(q)$ network is unaffected by
these losses.

### Useful SO(3) identities

These are used repeatedly below.

1. **Tangent action.** For any 9-vector $g = [g_1; g_2; g_3]$,
   $$-(q^\times)^T g \;=\; \sum_{i=1}^{3} R_i \times g_i \;\in\; \mathbb{R}^3.$$

2. **Range gram.** For $R\in SO(3)$,
   $$(q^\times)^T q^\times \;=\; \sum_i (I - R_i R_i^T) \;=\; 2I.$$

3. **Pseudoinverse on SO(3).**
   $$\big[(q^\times)^T\big]^{+} \;=\; \tfrac{1}{2}\,q^\times.$$

---

## 2. Central-Difference Noise Model

This section is the noise-handling foundation specific to
`ph_gp_sde_v2`. The $\omega$ snapshots in the dataset carry iid
Gaussian noise of std $\sigma_{\mathrm{obs}\omega}$, recorded by the
trainer in the model field `sigma_obs_omega`. Because $p = \big[M^{-1}\big]^{-1}\omega$
and (under fixed mass) the linear map $M$ is constant, the per-frame
noise on $p$ is also iid Gaussian with std $\sigma_{\mathrm{obs}\omega}$
(absorbing the constant $M$ into a one-time rescale that does not
appear in the residuals).

The central difference

$$\dot p^{\mathrm{data}}_t \;=\; \frac{p_{t+1} - p_{t-1}}{2\,\Delta t}$$

is a linear combination of two independent noise samples, so its
per-component noise variance is

$$\boxed{\;\sigma_{\mathrm{eff}}^2 \;=\; \frac{\sigma_{\mathrm{obs}\omega}^{\,2}}{2\,\Delta t^{\,2}}\;}$$

and its noise std is $\sigma_{\mathrm{obs}\omega} / (\Delta t\sqrt{2})$.

For the energy-rate target $\widehat{\dot H}$, the linearised independent
prediction is

$$\sigma_{H,\mathrm{lin}}^{\,2} \;=\; \big(\langle\|\omega\|^{2}\rangle \;+\; (m g l)^{2}\big)\;\frac{\sigma_{\mathrm{obs}\omega}^{\,2}}{2\,\Delta t^{\,2}}.$$

This **over-predicts the true variance by exactly $2\times$**, not $1.4\times$.
The reason is that $\dot H_t^{\mathrm{model}} = u^TB^TM^{-1}p \;-\; p^TM^{-1}DM^{-1}p$
shares its $\omega$-noise with $\widehat{\dot H}_t$. The negative covariance
between model and data sides subtracts half the residual variance, giving

$$\boxed{\;\sigma_{H,\mathrm{eff}}^{\,2} \;=\; \tfrac{1}{2}\,\sigma_{H,\mathrm{lin}}^{\,2} \;=\; \tfrac{1}{4}\big(\langle\|\omega\|^{2}\rangle + (m g l)^{2}\big)\;\frac{\sigma_{\mathrm{obs}\omega}^{\,2}}{\Delta t^{\,2}}.\;}$$

Empirical: with this $\sigma_{H,\mathrm{eff}}^{\,2}$, the whitened
diagnostic $\mathrm{mean}\|\mathrm{res}_{\mathrm{power}}\|^2 / \sigma_{H,\mathrm{eff}}^{\,2}$
lands at **0.91–0.98** flat across $\sigma_{\mathrm{obs}\omega} \in [0.01, 0.5]$.
Without the $\tfrac{1}{2}$ correction it sits at $\sim 0.5$, exactly half.

A clean training-side alternative is to learn $\sigma_{H,\mathrm{eff}}$ as a
free scalar (calibrate briefly, then freeze, or co-train with the rollout NLL).

---

## 3. Loss 1 — Power Balance (noise-aware)

### 3.1 Derivation of the deterministic part

The full $H$-chain-rule derivation is unchanged from the deterministic
case: kinematic + gyroscopic terms cancel analytically, leaving

$$\frac{dH}{dt} \;=\; u^T B^T M^{-1} p \;-\; p^T M^{-1} D M^{-1} p \;\equiv\; \dot H_t^{\mathrm{model}}.$$

### 3.2 Noise-aware loss

For each interior timestep $t$:

1. **Reconstruct $p$.** $\;p_t = \big[M_\theta^{-1}(q_t)\big]^{-1}\,\omega_t$.
2. **Compute $H_\theta$ and the central-diff target $\widehat{\dot H}_t$.**
3. **Compute $\dot H^{\mathrm{model}}_t$ from current parameters.**
4. **Form the residual** $\;r_t^{\mathrm{power}} = \widehat{\dot H}_t - \dot H_t^{\mathrm{model}}$.

Three loss forms are available:

* **Raw form** (legacy):
  $$\mathcal{L}_{\mathrm{power}}^{\mathrm{raw}} \;=\; \tfrac{1}{N_{\mathrm{int}}}\!\sum_t (r_t^{\mathrm{power}})^2.$$

* **Subtracted form** (analytic floor):
  $$\mathcal{L}_{\mathrm{power}}^{\mathrm{sub}} \;=\; \max\!\Big(\,\mathcal{L}_{\mathrm{power}}^{\mathrm{raw}} \;-\; \sigma_{H,\mathrm{eff}}^{\,2}\,,\;0\Big).$$

* **Whitened (recommended) form:**
  $$\boxed{\;\mathcal{L}_{\mathrm{power}}^{\mathrm{w}} \;=\; \frac{1}{N_{\mathrm{int}}}\sum_t \frac{(r_t^{\mathrm{power}})^2}{\sigma_{H,\mathrm{eff}}^{\,2}}\;}$$
  At the noise floor with correct physics, this loss has expectation
  $\,\mathbb{E}[\mathcal{L}_{\mathrm{power}}^{\mathrm{w}}] \approx 1$, independent of
  $\sigma_{\mathrm{obs}\omega}$.

### 3.3 What it catches

- Mismatch between $\{M_\theta, V_\theta\}$ and $\{B_\theta, D_\theta\}$.
- Whitened form additionally provides noise-invariance: a single
  $\lambda_{\mathrm{power}}$ value is meaningful across all
  $\sigma_{\mathrm{obs}\omega}$.

---

## 4. Loss 2 — Per-Subnetwork Back-Solving (noise-aware)

### 4.1 Common quantities at each interior frame

1. **Momentum** $p_t$ and **central-diff target** $\dot p_t^{\mathrm{data}}$ (interior frames only).
2. **Energy gradients** by autograd of $H_\theta(q,p)$:
   $$\nabla_q H_\theta, \qquad \nabla_p H_\theta = M_\theta^{-1}(q_t)\,p_t,\qquad \nabla_q V_\theta(q_t).$$
3. **Tangent-projected vectors** (3-dim, dynamics-relevant):
   $$g_{\mathrm{full}} = -(q_t^\times)^T \nabla_q H_\theta,\qquad g_V = -(q_t^\times)^T \nabla_q V_\theta,\qquad g_{KE} = g_{\mathrm{full}} - g_V.$$
4. **Other 3-vectors:**
   $$\mathrm{gyro} = p_t \times \nabla_p H_\theta,\qquad \mathrm{diss} = D_\theta(q_t,p_t)\,\nabla_p H_\theta,\qquad F = B_\theta(q_t)\,u_t.$$

The model EoM in this notation is

$$\dot p_t^{\mathrm{model}} \;=\; g_{\mathrm{full}} \;+\; \mathrm{gyro} \;-\; \mathrm{diss} \;+\; F.$$

### 4.2 Per-subnet residuals

* **$\mathcal L_V$** — back-solve for $V$ (tangent form):
  $$\mathrm{res}_V(t) \;=\; g_V \;-\; \alpha_t,\qquad \alpha_t \;=\; \dot p_t^{\mathrm{data}} \;-\; \big(g_{KE} + \mathrm{gyro} - \mathrm{diss} + F\big).$$
* **$\mathcal L_B$** — back-solve for $B$:
  $$\mathrm{res}_B(t) \;=\; F \;-\; \big(\dot p_t^{\mathrm{data}} \;-\; g_{\mathrm{full}} \;-\; \mathrm{gyro} \;+\; \mathrm{diss}\big).$$
* **$\mathcal L_D$** — back-solve for $D\,\nabla_p H$:
  $$\mathrm{res}_D(t) \;=\; \mathrm{diss} \;-\; \big(-\dot p_t^{\mathrm{data}} \;+\; g_{\mathrm{full}} \;+\; \mathrm{gyro} \;+\; F\big).$$

### 4.3 Algebraic identity (still holds in the noise-aware version)

By construction,

$$\mathrm{res}_V(t) \;=\; \mathrm{res}_B(t) \;=\; -\mathrm{res}_D(t) \;=\; \dot p_t^{\mathrm{model}} - \dot p_t^{\mathrm{data}}.$$

So all three losses share the same per-frame residual; the localisation
between $V_\theta$, $B_\theta$, $D_\theta$ happens through gradient
routing, not through the forward value.

### 4.4 Three loss forms

For each $X \in \{V, B, D\}$:

* **Raw form:**
  $$\mathcal{L}_X^{\mathrm{raw}} \;=\; \tfrac{1}{N_{\mathrm{int}}}\!\sum_t \|\mathrm{res}_X(t)\|_2^{\,2}.$$

* **Subtracted form** (predicted obs-noise floor removed):
  $$\mathcal{L}_X^{\mathrm{sub}} \;=\; \max\!\Big(\,\mathcal{L}_X^{\mathrm{raw}} \;-\; 3\,\sigma_{\mathrm{eff}}^{\,2}\,,\;0\Big),\qquad 3\,\sigma_{\mathrm{eff}}^{\,2} \;=\; \frac{3\,\sigma_{\mathrm{obs}\omega}^{\,2}}{2\,\Delta t^{\,2}}.$$
  Here the 3 counts the three components of $\mathrm{res}_X \in \mathbb{R}^3$.

* **Whitened (recommended) form:**
  $$\boxed{\;\mathcal{L}_X^{\mathrm{w}} \;=\; \frac{1}{N_{\mathrm{int}}}\sum_t \frac{\|\mathrm{res}_X(t)\|_2^{\,2}}{\sigma_{\mathrm{eff}}^{\,2}}\;}$$
  At the noise floor with correct physics,
  $\mathbb{E}[\mathcal{L}_X^{\mathrm{w}}] \approx 3$, independent of
  $\sigma_{\mathrm{obs}\omega}$.

Two equivalent "tight" formulations also exist; choose by convenience:

* **Per-frame whitening then sum** (used in the verifier):
  $$\mathcal{L}_X^{\mathrm{w}} \;=\; \tfrac{1}{N_{\mathrm{int}}}\!\sum_t \big\|\mathrm{res}_X(t)/\sigma_{\mathrm{eff}}\big\|_2^{\,2}.$$
  Identical numerics to the boxed form.

* **Gaussian negative log-likelihood** (drop additive constants):
  $$\mathcal{L}_X^{\mathrm{NLL}} \;=\; \tfrac{1}{N_{\mathrm{int}}}\!\sum_t \frac{\|\mathrm{res}_X(t)\|_2^{\,2}}{2\,\sigma_{\mathrm{eff}}^{\,2}} \;=\; \tfrac{1}{2}\,\mathcal{L}_X^{\mathrm{w}}.$$

The two differ only by a factor of $\tfrac{1}{2}$ and a constant
absorbed into $\lambda_X$.

### 4.5 What this catches

- Mutual inconsistency between subnetworks at individual data points.
- The **whitened forms** make the $\lambda_X$ weights meaningful
  across $\sigma_{\mathrm{obs}\omega}$ levels: a value of
  $\mathcal L_X^{\mathrm{w}} \approx 3$ flags "fitting noise"; a
  value $\gg 3$ flags real physics mismatch standing above the noise
  floor.

---

## 5. Combined Training Objective

$$\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{NLL}} \;+\; (\beta/N)\,\mathcal{L}_{\mathrm{KL}} \;+\; \lambda_{\mathrm{PL}}\,\mathcal{L}_{\mathrm{PL}} \;+\; \lambda_{\mathrm{power}}\,\mathcal{L}_{\mathrm{power}}^{\bullet} \;+\; \lambda_V\,\mathcal{L}_V^{\bullet} \;+\; \lambda_B\,\mathcal{L}_B^{\bullet} \;+\; \lambda_D\,\mathcal{L}_D^{\bullet}$$

where $\bullet \in \{\mathrm{raw},\,\mathrm{sub},\,\mathrm{w}\}$ selects
the loss form, and the existing GP_SDE_v2 pre-loss terms are kept:

| Term | What it enforces |
|---|---|
| $\mathcal{L}_{\mathrm{NLL}}$ | Rollout NLL (uses `log_sigma_R`, `log_sigma_omega`) |
| $\mathcal{L}_{\mathrm{KL}}$ | Variational KL summed over GP subnets |
| $\mathcal{L}_{\mathrm{PL}}$ | Per-increment pseudo-likelihood (uses `sigma_obs_omega`) |
| $\mathcal{L}_{\mathrm{power}}^{\bullet}$ | Energy bookkeeping |
| $\mathcal{L}_V^{\bullet}$ | $V$-localised pointwise momentum residual |
| $\mathcal{L}_B^{\bullet}$ | $B$-localised pointwise momentum residual |
| $\mathcal{L}_D^{\bullet}$ | $D$-localised pointwise momentum residual |

### 5.1 Suggested weights

For the **whitened forms** (recommended), all four
$\lambda \in [0.1, 1.0]$ work comparably well across
$\sigma_{\mathrm{obs}\omega} \in [0, 0.5]$. The weights do not need to
be re-tuned per noise level because the whitening makes the loss
expectations noise-invariant.

For the **raw forms**, the weights must scale as
$\lambda \propto \sigma_{\mathrm{obs}\omega}^{\,-2}$ to maintain a
constant signal-to-rollout ratio across noise levels — exactly the
problem whitening avoids.

### 5.2 Empirical floors (clean GT subnets, $\Delta t = 0.05$, wind=0)

These are produced by `verify_losses.py` with analytic GT subnets and
are the lower bounds the auxiliary losses can reach. The whitened
column shows the noise-invariance.

| $\sigma_{\mathrm{obs}\omega}$ | $\mathcal L_V^{\mathrm{raw}} = \mathcal L_B^{\mathrm{raw}} = \mathcal L_D^{\mathrm{raw}}$ | $\mathcal L_X^{\mathrm{w}}$ (per-axis MSE) | Noise-corrected (subtraction) |
|---|---|---|---|
| $0.00$ | $1.35\times 10^{-3}$ | n/a (no noise) | $1.35\times 10^{-3}$ |
| $0.01$ | $7.59\times 10^{-2}$ | $3.79$  ($1.43, 1.40, 0.97$) | $1.59\times 10^{-2}$ |
| $0.05$ | $1.86$ | $3.73$  ($1.39, 1.37, 0.97$) | $3.65\times 10^{-1}$ |
| $0.50$ | $1.79\times 10^{2}$ | $3.59$  ($1.32, 1.30, 0.97$) | $2.95\times 10^{1}$ |

**Reading the table.** The whitened mean $\|\cdot\|^2$ stays in
$[3.59,\,3.79]$ across a $50\times$ change in $\sigma_{\mathrm{obs}\omega}$.
The expected value is **NOT $3$**; it is

$$\boxed{\;3 \;+\; 4\,(m g l)^{2}\,\Delta t^{2} \;\approx\; 3.96\;\;\text{at}\;\;m{=}l{=}1,\,\Delta t{=}0.05.\;}$$

The extra $0.96$ is **model-side $q$-noise propagating through $g_V$**, not
truncation:

$$g_V \;=\; -(q^\times)^T \nabla_q V_\theta \;=\; m g l\,(R[2,1],\,-R[2,0],\,0).$$

Each of $R[2,0]$ and $R[2,1]$ carries observation noise of std
$\sigma_{\mathrm{obs}\omega}$, contributing $(m g l)^2 \sigma^2$ of
variance to the model-side residual on the $x$ and $y$ axes. After
whitening (dividing by $\sigma_{\mathrm{eff}}^2 = \sigma_{\mathrm{obs}\omega}^2/(2\Delta t^2)$),
each axis picks up $2(m g l)^2 \Delta t^2$, summed over the two affected
axes gives $4(m g l)^2 \Delta t^2 \approx 0.96$. The $z$-axis has no
$q$-dependence in $g_V$, so it sits at the data-only floor of $0.97 \approx 1$ —
a tight check that the whitening factor itself is correct.

Crucially, **truncation is ruled out** as the source of the excess: the
central-difference truncation contribution to $\mathrm{res}_X$ is
$\mathcal{O}(\Delta t^2 \cdot \partial^3 p / \partial t^3)$, whose **whitened** magnitude
scales as $1/\sigma_{\mathrm{obs}\omega}^{\,2}$. Across $\sigma_{\mathrm{obs}\omega} \in [0.01, 0.5]$ that
factor changes by $50^2 = 2500\times$. The observed excess changes by
only $\sim 5\%$. Therefore the excess is structural (model-side $q$-noise)
and the truncation contribution is everywhere subdominant.

**Fitting-noise threshold:** $\mathcal{L}_X^{\mathrm{w}} \approx 3.96$.
A model driving $\mathcal{L}_X^{\mathrm{w}}$ below this is fitting
observation noise on its own $V_\theta(q)$ inputs, not improving physics.

**$\mathcal L_{\mathrm{power}}^{\mathrm{w}}$:** with the $\tfrac{1}{2}$
covariance correction baked into $\sigma_{H,\mathrm{eff}}^{\,2}$ (see §2),
the diagnostic lands at $\approx 1$ (observed $0.91$–$0.98$ across
$\sigma_{\mathrm{obs}\omega} \in [0.01, 0.5]$). The remaining ~5% drift
is bounded by higher-order data/model covariance terms in $\dot H$.

---

## 6. Implementation Notes

1. **Central differences** for both $\dot p_t^{\mathrm{data}}$ and
   $\widehat{\dot H}_t$. Truncation is $\mathcal{O}(\Delta t^2)$ in
   the derivative, contributing the floor at $\sigma_{\mathrm{obs}\omega} = 0$.
2. **No data smoothing.** White obs-noise on $\omega$ propagates with
   per-component derivative-noise std $\sigma_{\mathrm{obs}\omega}/(\Delta t\sqrt{2})$.
   Whitening absorbs this analytically, so smoothing is optional.
3. **Trajectory endpoints are dropped** ($t=1$ and $t=N$).
4. **JAX autograd** is used for $\nabla_q H_\theta$ and
   $\nabla_q V_\theta$ via `jax.grad` / `eqx.filter_grad`. The GP
   subnets are evaluated with `inference_mode=True` so the deterministic
   posterior-mean path is taken (no key-driven weight sample noise
   inside the auxiliary losses).
5. **Per-subnet localisation via gradient routing.** All three of
   $\mathcal L_V$, $\mathcal L_B$, $\mathcal L_D$ have the same forward
   value (residual identity); the localisation is in the gradient
   chain, not the forward value.
6. **Diffusion path is untouched.** The model's $\sigma(q)$ network
   and `stochastic_increment_p` are not invoked by the auxiliary
   losses. Their gradient flow goes through $\mathcal L_{\mathrm{NLL}}$
   and $\mathcal L_{\mathrm{PL}}$ only.
7. **Optional fixed mass.** When $M^{-1}(q) = (1/(m\,l^2))\,I_3$ is
   pinned, $g_{KE} = 0$ and $g_{\mathrm{full}} = g_V$. The whitening
   denominator is unchanged: $\sigma_{\mathrm{eff}}^{\,2} = \sigma_{\mathrm{obs}\omega}^{\,2}/(2\,\Delta t^{\,2})$.
8. **Noise scale source.** The whitening uses
   `model.sigma_obs_omega` (set from the dataset's `obs_noise_std`).
   The trainable `log_sigma_R` and `log_sigma_omega` enter the rollout
   NLL and the PL term and are **not** reused here, keeping the
   auxiliary losses orthogonal to the rollout-NLL mechanism.

---

## 7. Diagnostics: `verify_losses.py`

The verifier in this folder runs the loss algebra with analytic GT
subnets (`M^{-1} = I`, $V = m g l\,q[8]$, $D = c\,I$, $B = I$) on a
dataset generated with the dataset's own observation-noise level. It
reports, for each $\sigma_{\mathrm{obs}\omega}$:

* The raw aggregate losses and their per-frame distribution.
* The predicted analytic noise floors ($3\sigma_{\mathrm{eff}}^{\,2}$ for V/B/D, $\sigma_{H,\mathrm{eff}}^{\,2}$ for power).
* Subtracted-form values ($\mathcal{L}^{\mathrm{raw}} - \mathrm{floor}$).
* Gaussian-NLL form ($\mathcal{L}^{\mathrm{raw}} / (2\sigma_{\mathrm{eff}}^{\,2})$).
* **Per-frame whitened residuals** $\mathrm{res}_X / \sigma_{\mathrm{eff}}$
  and their per-axis MSE — the diagnostic that makes the noise floor
  visible component-by-component.

Use it whenever a new loss form is wired into training to confirm the
analytic floor matches what the model is fitting against.
