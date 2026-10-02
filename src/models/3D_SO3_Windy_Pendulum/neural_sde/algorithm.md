# Unstructured Neural SDE — Algorithm

A step-by-step mathematical specification of the model in
[`network.py`](network.py) and the training procedure in
[`train.py`](train.py). No port-Hamiltonian structure, no Lie-group
integrator — this is the SDE analogue of the reference
`UnstructuredSO3NODE` baseline.

---

## 1. State, control, and observed data

The 3D windy pendulum environment emits, at every snapshot, the 15-dim
tuple

$$
y \;=\; \big(\,\underbrace{\mathrm{vec}(R)}_{\in\,\mathbb{R}^{9}},\;
\underbrace{\omega}_{\in\,\mathbb{R}^{3}},\;
\underbrace{u}_{\in\,\mathbb{R}^{3}}\,\big) \;\in\; \mathbb{R}^{15}
$$

where

- $R \in \mathrm{SO}(3)$ — orientation of the rigid body, row-major
  flattened to length 9 — written $q := \mathrm{vec}(R)$ in what
  follows.
- $\omega \in \mathbb{R}^{3}$ — body-frame angular velocity.
- $u \in \mathbb{R}^{3}$ — control torque, *held constant* across each
  observation interval.

The integrated state is the 12-dim concatenation

$$
x \;=\; (q,\,\omega) \;\in\; \mathbb{R}^{12}.
$$

A trajectory of length $T$ for one initial condition and one control is
the time-stacked array $\,y_{0:T} \in \mathbb{R}^{T\times 15}$.

---

## 2. Continuous-time SDE the model defines

The model is a **fully unstructured Stratonovich SDE** on $\mathbb{R}^{12}$:

$$
\boxed{\;
\mathrm{d}x_t \;=\; f_\theta(x_t, u)\,\mathrm{d}t \;+\; \Sigma_\theta(x_t,u)\circ\mathrm{d}W_t,
\qquad
W_t \in \mathbb{R}^{3}.
\;}
$$

The "$\circ$" denotes Stratonovich integration; the Stratonovich
convention is used for the same reason as in the env's data generator
and in `ph_gp_sde` / `ph_nn_sde_debug` — it commutes with the
chain rule, which keeps the Heun-type predictor–corrector
discretisation unbiased even when $\Sigma_\theta$ depends on the
state.

The two learnable maps are

$$
\begin{aligned}
f_\theta\!\!: \mathbb{R}^{15} &\to \mathbb{R}^{12}, &\quad &\text{(drift / time derivative)}\\
\sigma_\theta\!\!: \mathbb{R}^{15} &\to \mathbb{R}^{3}_{\geq 0}, &\quad &\text{(diagonal diffusion scale on \(\omega\))}
\end{aligned}
$$

both implemented as 3-layer MLPs (input $(q, \omega, u) \in \mathbb{R}^{15}$,
hidden width $H$, $\tanh$ activations).

The diffusion matrix is concentrated on the angular-velocity block:

$$
\Sigma_\theta(x, u) \;=\;
\begin{bmatrix}
\mathbf{0}_{9\times 3} \\[2pt]
\mathrm{diag}\!\big(\sigma_\theta(x,u)\big)_{3\times 3}
\end{bmatrix}
\;\in\; \mathbb{R}^{12 \times 3}.
$$

Positivity of the diffusion scale is enforced with a softplus on the
network's raw output:

$$
\sigma_\theta(x,u) \;=\; \mathrm{softplus}\!\big(\mathrm{MLP}_{\sigma_\theta}(q,\omega,u)\big), \qquad \sigma \geq 0.
$$

Equivalently, in component form,

$$
\begin{aligned}
\mathrm{d}q_t       \;&=\; f_\theta(x_t, u)_{[1:9]}\,\mathrm{d}t \\
\mathrm{d}\omega_t  \;&=\; f_\theta(x_t, u)_{[10:12]}\,\mathrm{d}t \;+\; \sigma_\theta(x_t,u)\odot\,\mathrm{d}W_t \quad\text{(Stratonovich).}
\end{aligned}
$$

> No $\mathrm{SO}(3)$ projection is performed. The 9-dim $q$ is allowed
> to drift off the manifold during integration, exactly matching the
> "unstructured" baseline philosophy of the reference Neural ODE.

---

## 3. Discretisation: Stratonovich Heun on $\mathbb{R}^{12}$

A snapshot interval of length $\Delta t$ (set by the dataset's
`t_eval`) is subdivided into $N_{\text{sub}}$ substeps of size

$$
h \;=\; \frac{\Delta t}{N_{\text{sub}}}.
$$

For each substep $k = 0, \ldots, N_{\text{sub}}-1$, draw a single
independent Wiener increment

$$
\Delta W_k \;\sim\; \mathcal{N}\!\big(\mathbf{0},\, h\,I_3\big), \qquad \Delta W_k \in \mathbb{R}^{3},
$$

which is **reused** in both stages of the Heun predictor–corrector.
Define the diffusion-padded vector

$$
G_\theta(x, u; \Delta W) \;=\; \begin{bmatrix}\mathbf{0}_{9}\\[2pt]\sigma_\theta(x,u)\odot\Delta W\end{bmatrix} \;\in\; \mathbb{R}^{12}.
$$

**Stage 1 — drift / diffusion at the current state.**

$$
\begin{aligned}
\mathbf{f}_1 \;&=\; f_\theta(x_k,\,u),\\
\mathbf{g}_1 \;&=\; G_\theta(x_k,\,u;\,\Delta W_k).
\end{aligned}
$$

**Stage 2 — Euler predictor (same $\Delta W_k$).**

$$
\widetilde x_k \;=\; x_k \;+\; \mathbf{f}_1\,h \;+\; \mathbf{g}_1.
$$

**Stage 3 — re-evaluate at the predicted state ($\Delta W_k$ reused).**

$$
\begin{aligned}
\mathbf{f}_2 \;&=\; f_\theta(\widetilde x_k,\,u),\\
\mathbf{g}_2 \;&=\; G_\theta(\widetilde x_k,\,u;\,\Delta W_k).
\end{aligned}
$$

**Stage 4 — corrector (average drift + average diffusion).**

$$
\boxed{\;
x_{k+1} \;=\; x_k \;+\; \tfrac{1}{2}(\mathbf{f}_1 + \mathbf{f}_2)\,h \;+\; \tfrac{1}{2}(\mathbf{g}_1 + \mathbf{g}_2).
\;}
$$

These four stages are exactly what the `step` method computes.

> **Why reuse $\Delta W_k$?** A predictor–corrector scheme with a
> *fresh* second draw would converge to the Itô solution; reusing the
> same $\Delta W_k$ in the predictor and corrector evaluations
> reproduces the **Stratonovich** chain rule and gives strong order
> $1/2$, weak order $1$ for general state-dependent diffusions
> (Kloeden–Platen, "Heun" / "stochastic trapezoidal" scheme). It also
> matches the integrator the env uses to *generate* the data, so model
> and target speak the same SDE convention.

---

## 4. Rollout over an observation window

Let $T_{\mathrm{obs}}$ be the number of snapshots per training window
(`num_points`), and let $N_{\mathrm{outer}} = T_{\mathrm{obs}} - 1$ be
the number of *outer* intervals between snapshots.

Given an initial $x_0 \in \mathbb{R}^{12}$, a constant control
$u \in \mathbb{R}^{3}$, and a pre-sampled increment tensor

$$
\big\{\Delta W_{i,k}\big\}_{\,i=0,\ldots,N_{\mathrm{outer}}-1\,;\, k=0,\ldots,N_{\mathrm{sub}}-1}, \qquad \Delta W_{i,k} \in \mathbb{R}^{3},
$$

the rollout proceeds:

1. **Outer loop** $i = 0, 1, \ldots, N_{\mathrm{outer}}-1$:
   - Initialise $x \gets x_i$ (state at snapshot $i$).
   - **Inner loop** $k = 0, \ldots, N_{\mathrm{sub}}-1$ (one Stratonovich Heun substep, $\Delta W_{i,k}$ reused across the two stages):

     $$
     \begin{aligned}
     \mathbf{f}_1,\,\mathbf{g}_1 \;&=\; f_\theta(x,u),\;G_\theta(x,u;\Delta W_{i,k}),\\
     \widetilde x \;&=\; x + \mathbf{f}_1\,h + \mathbf{g}_1,\\
     \mathbf{f}_2,\,\mathbf{g}_2 \;&=\; f_\theta(\widetilde x,u),\;G_\theta(\widetilde x,u;\Delta W_{i,k}),\\
     x \;&\gets\; x + \tfrac{1}{2}(\mathbf{f}_1+\mathbf{f}_2)\,h + \tfrac{1}{2}(\mathbf{g}_1+\mathbf{g}_2).
     \end{aligned}
     $$

   - Record $x_{i+1} \gets x$.

2. Output the trajectory

$$
\hat{x}_{0:N_{\mathrm{outer}}} \;\in\; \mathbb{R}^{(N_{\mathrm{outer}}+1)\times 12}.
$$

Padding $\hat{x}$ with the constant control $u$ along the last axis
yields the 15-dim form $\hat{y} \in \mathbb{R}^{(N_{\mathrm{outer}}+1)\times 15}$
that matches the env observations and the loss helpers.

---

## 5. Training objective

Let the batch contain $B$ trajectories
$y^{(b)}_{0:T} \in \mathbb{R}^{T\times 15}$, $b = 1,\ldots,B$, all of
length $T = T_{\mathrm{obs}}$. Each batch element is rolled forward
from its own observed initial state

$$
x^{(b)}_0 \;=\; y^{(b)}_{0,\,1:12}, \qquad u^{(b)} \;=\; y^{(b)}_{0,\,13:15},
$$

using freshly sampled Wiener increments. The training loss is the
hybrid $\mathrm{SO}(3)$ + $\mathbb{R}^{3+3}$ residual

$$
\mathcal{L}(\theta) \;=\; \mathcal{L}_{\mathrm{geo}}(\theta) \;+\; \mathcal{L}_{L^2}(\theta),
$$

where, summed/averaged over snapshots $t = 1, \ldots, T-1$ and batch
elements $b$,

$$
\begin{aligned}
\mathcal{L}_{\mathrm{geo}}(\theta) \;&=\;
\frac{1}{B(T-1)}\sum_{b,t} \theta_{b,t}^2,
\qquad
\theta_{b,t} \;=\; \mathrm{atan2}\!\Big(\big\lVert\mathrm{vee}\big(\tfrac{M-M^{\!\top}}{2}\big)\big\rVert_2,\,\tfrac{\mathrm{tr}\,M-1}{2}\Big),\\
&\hspace{12em} M \;=\; R^{(b)}_t\,\big(\hat{R}^{(b)}_t\big)^{\!\top},\\[6pt]
\mathcal{L}_{L^2}(\theta) \;&=\;
\frac{1}{B(T-1)}\sum_{b,t}
\Big\lVert\,\big(\omega^{(b)}_t,\,u^{(b)}\big) \;-\; \big(\hat\omega^{(b)}_t,\,u^{(b)}\big)\,\Big\rVert_2^2.
\end{aligned}
$$

The first term is the squared-geodesic distance on $\mathrm{SO}(3)$
(`compute_geodesic_loss_safe`, fp32-stable two-argument `atan2` form);
the second is a vanilla MSE on the angular-velocity and control
channels. Because the predicted $\hat R$ drifts off $\mathrm{SO}(3)$, the
geodesic helper computes $M$ from the raw $\hat q$ reshaped to a
$3\times 3$ matrix without orthogonalisation — the `_safe` variants
guard against the small-angle and antipodal singularities.

> **Two independent SDE samples.** Each training step samples a
> *fresh* $\Delta W$ batch — different from the env trajectory's path,
> which was generated under a different RNG. So the loss compares
> two independent draws from two different SDEs (model vs env). The
> drift learns the env's *mean* dynamics; the diffusion's only data
> signal is whatever residual variance the rollout MSE explains by
> tilting $\sigma_\theta$. This is the same training regime as
> `ph_nn_sde_debug` — for a stronger $\sigma$ signal you'd add a
> per-increment pseudo-likelihood term, as `ph_gp_sde` does.

---

## 6. Optimisation

Parameters $\theta$ contain only the MLP weights of $f_\theta$ and
$\sigma_\theta$. The optimiser is

$$
\theta_{n+1} \;=\; \theta_n \;-\; \mathrm{AdamW}\!\Big(\,\mathrm{clip}_{\lVert\cdot\rVert\le c}\!\big(\nabla_\theta \mathcal{L}\big)\,;\;\eta,\,\lambda\Big),
$$

with global-norm gradient clip $c$ (`--grad_clip`, default 1.0),
learning rate $\eta$ (`--learn_rate`, default $10^{-3}$), and weight
decay $\lambda = 10^{-4}$. Gradients flow through the entire
Stratonovich Heun unroll: backprop differentiates the discretised
trajectory $\hat{x}^{(b)}_{0:N_{\mathrm{outer}}}$ end-to-end with
respect to $\theta$, **including both predictor and corrector stages
of every substep** (each substep contains four network evaluations:
$f_\theta$ and $\sigma_\theta$ at $x_k$ and again at $\widetilde x_k$).
The noise itself, $\Delta W_{i,k}$, is treated as a fixed constant
inside the unrolled graph for that step — only $f_\theta$ and
$\sigma_\theta$ are differentiated.

---

## 7. End-to-end training step (recap)

Putting steps 3–6 together, one training iteration is:

1. **Sample a Wiener tensor**

  $$\Delta W \in \mathbb{R}^{B\times N_{\mathrm{outer}} \times N_{\mathrm{sub}} \times 3},
  \qquad \Delta W_{b,i,k} \sim \mathcal{N}(0, h\,I_3).$$

2. **Roll out the SDE** for every $b$ in parallel (vmap over the batch
  axis), starting from $x^{(b)}_0$ and using $u^{(b)}$ as the constant
  control. Pad with $u^{(b)}$ to recover the 15-dim form.

3. **Compute** $\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{geo}} + \mathcal{L}_{L^2}$ on
  $t = 1,\ldots,T-1$.

4. **Backprop, clip, and AdamW-step.**

5. **Periodically** (every `--eval_every` steps) re-roll the test
  batch with a fresh $\Delta W$ and log $(\mathcal{L}, \mathcal{L}_{\mathrm{geo}}, \mathcal{L}_{L^2})$;
  serialise a checkpoint and the running stats dict.

---

## 8. Final trajectory evaluation

After training finishes, for each control $u$ in
$\{(0,0,0),(\pm 1,\pm 1,\pm 1),(\pm 2,\pm 2,\pm 2)\}$ (the dataset's
control alphabet), the model is rolled out for the **full** environment
horizon $T$ (not the windowed $T_{\mathrm{obs}}$) using one fresh
$\Delta W$ sample per batch element. Per-trajectory totals

$$
\mathcal{L}^{\mathrm{traj}}_{b} \;=\; \sum_{t=1}^{T-1}\Big(\theta_{b,t}^2 \;+\; \big\lVert(\omega_{b,t},u_b)-(\hat\omega_{b,t},u_b)\big\rVert^2\Big)
$$

are reported as mean ± std over $b$. The full predicted trajectories
$\hat y$ are stashed in the stats dict (`train_x_hat`, `test_x_hat`)
so the comparison-PDF script can plot them next to `ph_gp_sde`,
`ph_nn_ode_v2`, etc., on the same axes.

---

## 9. What is *not* modelled (vs `ph_gp_sde`)

To make the contrast with the structured baseline crisp:

| Component                            | `ph_gp_sde`                               | `neural_sde` (this model)        |
|--------------------------------------|--------------------------------------------|----------------------------------|
| Inverse mass $M^{-1}(q)$           | learned PSD-GP subnet                      | absorbed into $f_\theta$         |
| Potential $V(q)$                   | learned scalar GP                          | absorbed into $f_\theta$         |
| Dissipation $D_w(q)$               | learned PSD-GP subnet                      | absorbed into $f_\theta$         |
| Control coupling $g(q)$            | learned $3\times 3$ GP-MatrixNet           | absorbed into $f_\theta$         |
| Drift form                           | $\dot p = p\times M^{-1}p + \sum_i R_i\times\partial_{q_i}H - D_w M^{-1}p + g\,u$ | free MLP on $(q,\omega,u)$ |
| Diffusion form                       | $\mathrm{d}p_{\text{stoch}} = R^{\!\top}\big(\ell\,R e_z \times \sigma(q)\,\mathrm{d}W\big)$ | $\mathrm{diag}(\sigma_\theta)\,\mathrm{d}W$ |
| Manifold preservation                | Lie–Heun (Stratonovich) on $\mathrm{SO}(3)\times\mathbb{R}^3$ | Stratonovich Heun on flat $\mathbb{R}^{12}$ |
| Variational weights / KL / ELBO      | yes (mean-field over RFF features)         | no                               |
| Per-increment pseudo-likelihood      | yes (anti-collapse for $\sigma$)           | no                               |

The neural SDE here is therefore the *minimal* black-box stochastic
analogue: it has the same I/O contract as `ph_gp_sde` (same dataset,
same loss-key shape, same checkpoint pattern) but absolutely no
physics inductive bias — useful as a control for ablation studies
that quantify how much the port-Hamiltonian structure and the
GP/ELBO machinery actually buy.
