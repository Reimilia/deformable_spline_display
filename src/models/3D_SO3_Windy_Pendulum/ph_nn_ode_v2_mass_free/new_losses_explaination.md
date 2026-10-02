# Port-Hamiltonian Loss Functions — Math Reference

This document describes the physics-informed losses currently active in
training. There are **four** auxiliary losses on top of the rollout loss:
`L_power`, `L_V`, `L_B`, `L_D`. The full pointwise momentum-residual loss
of Loss 2 is NOT used; it is replaced by three per-subnetwork
back-solving losses.

---

## 1. Equations of Motion

### Configuration dynamics

$$\dot{q} \;=\; q^\times \, \nabla_p H$$

### Momentum dynamics

$$\dot{p} \;=\; -(q^\times)^T \nabla_q H \;+\; p^\times \nabla_p H \;-\; D(q,p)\,\nabla_p H \;+\; B(q)\,u$$

### Symbols

| Symbol | Shape | Meaning |
|---|---|---|
| $q$ | $9$ | Vectorized rotation matrix (rows of $R$ stacked: $q = [R_1; R_2; R_3]$) |
| $p$ | $3$ | Angular momentum |
| $\omega = M^{-1}(q)\,p$ | $3$ | Angular velocity ($\omega \equiv \dot{q}$-conjugate body-frame) |
| $H(q,p) = \tfrac{1}{2} p^T M^{-1}(q)\,p + V(q)$ | scalar | Total energy |
| $\nabla_p H$ | $3$ | $= M^{-1}p = \omega$ |
| $\nabla_q H$ | $9$ | Gradient of $H$ w.r.t. configuration |
| $q^\times$ | $9\times 3$ | Kinematic map; $q^\times \omega$ stacks $R_i \times \omega$ |
| $p^\times$ | $3\times 3$ | Skew-symmetric of $p$ |
| $M(q)$ | $3\times 3$ PSD | Generalised mass / inertia |
| $D(q,p)$ | $3\times 3$ PSD | Dissipation matrix |
| $B(q)$ | $3\times m$ | Input matrix |
| $u$ | $m$ | Control input |

### Convention used by the model

The mass subnetwork outputs $M^{-1}(q)$ (not $M$). Wherever the math
below shows $M^{-1}$, that is what is evaluated directly. Momentum is
recovered by

$$p_t \;=\; \big[M_\theta^{-1}(q_t)\big]^{-1}\,\omega_t.$$

For the spherical pendulum this is the constant scalar $m\,l^2$, optionally
pinned to that value.

### Physical meaning of each $\dot{p}$ term

| Term | Role |
|---|---|
| $-(q^\times)^T \nabla_q H$ | Configuration-dependent force: gravity + KE-pose coupling |
| $p^\times \nabla_p H$ | Gyroscopic / Coriolis ($p\times\omega$); energy-conserving |
| $-D \nabla_p H$ | Dissipation; always removes energy |
| $B u$ | Control input; injects/extracts energy |

### Useful SO(3) identities

These are used repeatedly below.

1. **Tangent action.** For any 9-vector $g = [g_1; g_2; g_3]$,
   $$-(q^\times)^T g \;=\; \sum_{i=1}^{3} R_i \times g_i \;\in\; \mathbb{R}^3.$$

2. **Range gram.** For $R\in SO(3)$,
   $$(q^\times)^T q^\times \;=\; \sum_i \widehat{R_i}^T \widehat{R_i} \;=\; \sum_i (I - R_i R_i^T) \;=\; 3I - I \;=\; 2I.$$

3. **Pseudoinverse on SO(3).**
   $$\big[(q^\times)^T\big]^{+} \;=\; (q^\times)\,\big[(q^\times)^T q^\times\big]^{-1} \;=\; \tfrac{1}{2}\,q^\times.$$

---

## 2. Loss 1 — Power Balance

### 2.1 Derivation

Start from the chain rule:

$$\frac{dH}{dt} \;=\; (\nabla_q H)^T \dot{q} \;+\; (\nabla_p H)^T \dot{p}.$$

Substitute the EoM and group terms:

$$\frac{dH}{dt} \;=\; \underbrace{(\nabla_q H)^T\,q^\times \nabla_p H \;-\; (\nabla_p H)^T\,(q^\times)^T \nabla_q H}_{=\,0\text{ (transpose identity)}} \;+\; \underbrace{(\nabla_p H)^T\,p^\times\,\nabla_p H}_{=\,0\text{ (skew)}} \;-\;(\nabla_p H)^T D \nabla_p H \;+\; (\nabla_p H)^T B u.$$

What survives:

$$\boxed{\;\frac{dH}{dt} \;=\; u^T B^T\,M^{-1} p \;-\; p^T M^{-1}\,D\,M^{-1} p \;}$$

The first term is **power injected by control** (any sign), the second
is **power dissipated** ($\geq 0$).

### 2.2 Discrete loss

For each interior timestep $t \in \{2,\dots,N-1\}$ in a window
$\{(q_t,\omega_t,u_t)\}$:

1. **Reconstruct momentum.**
   $$p_t \;=\; \big[M_\theta^{-1}(q_t)\big]^{-1}\,\omega_t$$

2. **Energy.**
   $$H_\theta(q_t, p_t) \;=\; \tfrac{1}{2}\, p_t^T M_\theta^{-1}(q_t)\,p_t \;+\; V_\theta(q_t)$$

3. **LHS** (numerical, central difference):
   $$\widehat{\dot{H}}_t \;=\; \frac{H_\theta(q_{t+1}, p_{t+1}) \;-\; H_\theta(q_{t-1}, p_{t-1})}{2\,\Delta t}$$

4. **RHS** (analytic from current parameters):
   $$\dot{H}_t^{\mathrm{model}} \;=\; u_t^T B_\theta^T(q_t)\,\big[M_\theta^{-1}(q_t)\,p_t\big] \;-\; p_t^T M_\theta^{-1}(q_t)\,D_\theta(q_t,p_t)\,M_\theta^{-1}(q_t)\,p_t$$

5. **Loss.**
   $$\boxed{\;\mathcal{L}_{\text{power}} \;=\; \frac{1}{N_{\mathrm{int}}}\sum_{t} \big(\widehat{\dot{H}}_t \;-\; \dot{H}_t^{\mathrm{model}}\big)^2\;}$$

### 2.3 What it catches

- Mismatch between $\{M_\theta, V_\theta\}$ (which fix $H$) and
  $\{B_\theta, D_\theta\}$ (which fix $\dot H$).
- The kinematic and gyroscopic terms have already cancelled
  analytically, so this loss isolates the **energy budget** only.

---

## 3. Loss 2 — Per-Subnetwork Back-Solving

The full pointwise residual $\|\dot{p}_t^{\mathrm{data}} - \dot{p}_t^{\mathrm{model}}\|^2$
is not used. Instead, the EoM is rearranged three times, isolating one
subnetwork at a time. The three losses share the data-side quantity

$$\dot{p}_t^{\mathrm{data}} \;=\; \frac{p_{t+1} - p_{t-1}}{2\,\Delta t}, \qquad p_\tau \;=\; \big[M_\theta^{-1}(q_\tau)\big]^{-1}\,\omega_\tau,$$

with $p_t^{\mathrm{data}}$ detached (it is a target, not a model output).
$M^{-1}$ appears in three of the four EoM terms and cannot be cleanly
isolated, so there is no $\mathcal{L}_M$.

### 3.1 Common quantities at each interior frame

1. **Momentum** $p_t$ (as above; central-diff endpoints dropped).
2. **Energy gradients** by autograd of $H_\theta(q,p) = \tfrac{1}{2} p^T M_\theta^{-1}(q) p + V_\theta(q)$:
   $$\nabla_q H_\theta, \qquad \nabla_p H_\theta \;=\; M_\theta^{-1}(q_t)\,p_t.$$
   The $V$-only piece is also computed:
   $$\nabla_q V_\theta(q_t).$$
   By construction, $\nabla_q H_\theta = \nabla_q V_\theta + \nabla_q(\tfrac{1}{2} p^T M_\theta^{-1} p)$.
3. **Tangent-projected vectors** (3-dim, dynamics-relevant):
   $$g_{\mathrm{full}} \;=\; -(q_t^\times)^T \nabla_q H_\theta \;=\; \sum_i R_i \times (\nabla_q H_\theta)_i$$
   $$g_V \;=\; -(q_t^\times)^T \nabla_q V_\theta \;=\; \sum_i R_i \times (\nabla_q V_\theta)_i$$
   $$g_{KE} \;=\; g_{\mathrm{full}} - g_V \;=\; -(q_t^\times)^T \nabla_q (\tfrac{1}{2} p^T M_\theta^{-1} p)\cdot 2 \cdot \tfrac{1}{2}$$
4. **Other 3-vectors:**
   $$\mathrm{gyro} \;=\; p_t \times \nabla_p H_\theta, \qquad \mathrm{diss} \;=\; D_\theta(q_t,p_t)\,\nabla_p H_\theta, \qquad F \;=\; B_\theta(q_t)\,u_t.$$

The model EoM in this notation is

$$\dot{p}_t^{\mathrm{model}} \;=\; g_{\mathrm{full}} \;+\; \mathrm{gyro} \;-\; \mathrm{diss} \;+\; F.$$

### 3.2 $\mathcal{L}_V$ — back-solve for $V$ (tangent form)

#### Step 1 — Move all non-$V$ terms across the equality.

From the EoM,

$$-(q_t^\times)^T \nabla_q V_{\mathrm{implied}} \;=\; \dot{p}_t^{\mathrm{data}} \;-\; \big[\,g_{KE} \;+\; \mathrm{gyro} \;-\; \mathrm{diss} \;+\; F\,\big] \;=:\; \alpha_t \;\in\;\mathbb{R}^3.$$

This isolates the $V$ contribution on the LHS.

#### Step 2 — Two ways to compare $\nabla_q V_\theta$ to $\nabla_q V_{\mathrm{implied}}$.

* **9-dim (pseudoinverse) form.** Using identity (3), the minimum-norm
  reconstruction is
  $$\nabla_q V_{\mathrm{implied}} \;=\; -\big[(q_t^\times)^T\big]^{+}\alpha_t \;=\; -\tfrac{1}{2}\,q_t^\times \alpha_t \;=\; -\tfrac{1}{2}\,\big[R_1\times\alpha_t;\,R_2\times\alpha_t;\,R_3\times\alpha_t\big].$$
  The natural loss $\big\| \nabla_q V_\theta - \nabla_q V_{\mathrm{implied}} \big\|_2^2$
  has a **structural floor**: $\nabla_q V_\theta \in \mathbb{R}^9$ has both
  a tangent component (in $\mathrm{range}(q_t^\times)$, 3-dim) and a
  normal component (in $\ker((q_t^\times)^T)$, 6-dim). The pinv-based
  $\nabla_q V_{\mathrm{implied}}$ lives entirely in the tangent space, so
  the normal component of $\nabla_q V_\theta$ contributes a residual
  that does not affect dynamics and does not vanish even at ground truth.

* **Tangent (3-dim) form — used in code.** Project both sides through
  $-(q_t^\times)^T$ and compare in $\mathbb{R}^3$:
  $$\boxed{\;\mathcal{L}_V \;=\; \frac{1}{N_{\mathrm{int}}}\sum_t \big\|\,g_V(t) \;-\; \alpha_t\,\big\|_2^{\,2} \;}$$
  This is exactly the dynamics-relevant V error: at clean ground-truth
  data $\alpha_t = g_V(t)$ (since the EoM holds), so $\mathcal{L}_V \to 0$,
  with no kernel artefact. Forward value coincides numerically with
  $\mathcal{L}_B$ and $\mathcal{L}_D$ below; the gradient still routes
  selectively through $V_\theta$ and $M_\theta$ because $g_V$ and
  $g_{KE}$ are built from those subnets only.

### 3.3 $\mathcal{L}_B$ — back-solve for $B$

Move the $B$ term to one side:

$$B_\theta(q_t)\,u_t \;\stackrel{?}{=}\; \dot{p}_t^{\mathrm{data}} \;-\; g_{\mathrm{full}} \;-\; \mathrm{gyro} \;+\; \mathrm{diss}.$$

Loss:

$$\boxed{\;\mathcal{L}_B \;=\; \frac{1}{N_{\mathrm{int}}}\sum_t \big\| F \;-\; \big[\,\dot{p}_t^{\mathrm{data}} \;-\; g_{\mathrm{full}} \;-\; \mathrm{gyro} \;+\; \mathrm{diss}\,\big]\,\big\|_2^{\,2}.\;}$$

### 3.4 $\mathcal{L}_D$ — back-solve for $D \nabla_p H$

Move the $D$ term to one side:

$$D_\theta(q_t,p_t)\,\nabla_p H_\theta \;\stackrel{?}{=}\; -\dot{p}_t^{\mathrm{data}} \;+\; g_{\mathrm{full}} \;+\; \mathrm{gyro} \;+\; F.$$

Loss:

$$\boxed{\;\mathcal{L}_D \;=\; \frac{1}{N_{\mathrm{int}}}\sum_t \big\| \mathrm{diss} \;-\; \big[\,-\dot{p}_t^{\mathrm{data}} \;+\; g_{\mathrm{full}} \;+\; \mathrm{gyro} \;+\; F\,\big]\,\big\|_2^{\,2}.\;}$$

### 3.5 Algebraic relationships

These follow directly from the rearrangements above and are useful for
diagnostics:

1. The three residuals are equal up to sign:
   $$\mathrm{res}_V(t) \;=\; \mathrm{res}_B(t) \;=\; -\mathrm{res}_D(t) \;=\; \dot{p}_t^{\mathrm{model}} \;-\; \dot{p}_t^{\mathrm{data}}.$$
2. Hence the *forward values* satisfy
   $$\mathcal{L}_V \;\equiv\; \mathcal{L}_B \;\equiv\; \mathcal{L}_D \;\equiv\; \tfrac{1}{N_{\mathrm{int}}}\sum_t \big\|\dot{p}_t^{\mathrm{model}} - \dot{p}_t^{\mathrm{data}}\big\|_2^{\,2}.$$
3. The **gradients** differ. Each loss treats the named subnet as the
   source of error and routes its gradient accordingly through the autograd
   chains that build $g_V$ ($V_\theta, M_\theta$), $F$ ($B_\theta$),
   and $\mathrm{diss}$ ($D_\theta, M_\theta$).

### 3.6 What these losses catch

- Mutual inconsistency between subnetworks at individual data points
  (the rollout loss sees only integrated effects).
- Per-component decomposition localises which subnetwork is most to
  blame — only via the gradient routing, since the forward values are
  identical.

---

## 4. Combined Training Objective

$$\boxed{\;\mathcal{L}_{\mathrm{total}} \;=\; \mathcal{L}_{\mathrm{rollout}} \;+\; \lambda_{\mathrm{power}}\,\mathcal{L}_{\mathrm{power}} \;+\; \lambda_V\,\mathcal{L}_V \;+\; \lambda_B\,\mathcal{L}_B \;+\; \lambda_D\,\mathcal{L}_D \;}$$

| Term | What it enforces |
|---|---|
| $\mathcal{L}_{\mathrm{rollout}}$ | Trajectory matching ($L_2$ + geodesic on $R$) |
| $\mathcal{L}_{\mathrm{power}}$ | Energy bookkeeping (Loss 1) |
| $\mathcal{L}_V$ | $V$-localised pointwise momentum residual |
| $\mathcal{L}_B$ | $B$-localised pointwise momentum residual |
| $\mathcal{L}_D$ | $D$-localised pointwise momentum residual |

### 4.1 Suggested weights

- Default training: all four $\lambda \in [0.5, 1.0]$.
- Lower-noise data: weights can be increased; aux losses are then
  closer to ground truth and contribute more signal.
- Higher-noise data: weights should be decreased; central-difference
  noise on $\dot p^{\mathrm{data}}$ scales as $\sigma_{\mathrm{obs}}/\Delta t$
  and dominates these losses at high $\sigma_{\mathrm{obs}}$.

### 4.2 Empirical floors (clean GT subnets, $\Delta t = 0.05$)

These are the lower bounds the auxiliary losses can reach with perfect
physics, dominated by central-difference truncation and obs-noise
propagation:

| $\sigma_{\mathrm{obs}}$ | $\mathcal{L}_{\mathrm{power}}$ | $\mathcal{L}_V \!=\! \mathcal{L}_B \!=\! \mathcal{L}_D$ |
|---|---|---|
| $0$ | $\sim 1\times 10^{-3}$ (truncation) | $\sim 1\times 10^{-3}$ |
| $0.01$ | $\sim 1$ | $\sim 7.6\times 10^{-2}$ |
| $0.05$ | $\sim 26$ | $\sim 1.9$ |

Any training run that drives an aux loss below its floor is fitting
noise, not physics.

---

## 5. Implementation Notes

1. **Central differences** are used for both $\dot{p}_t^{\mathrm{data}}$
   and $\widehat{\dot H}_t$. Truncation is $\mathcal{O}(\Delta t^2)$ in
   the derivative; squared this gives the $\sim 10^{-3}$ floor at
   $\sigma_{\mathrm{obs}}=0$.
2. **No smoothing** is applied to $\dot{p}_t^{\mathrm{data}}$. White
   obs-noise of std $\sigma$ on $q$ yields derivative-noise std
   $\approx \sigma/(\Delta t\sqrt{2})$ per component.
3. **Trajectory endpoints are dropped** ($t=1$ and $t=N$) — the central
   difference is undefined there.
4. **Autograd with create_graph=True** is used for $\nabla_q H_\theta$
   and $\nabla_q V_\theta$, so gradients propagate during backprop.
5. **Cat-split decoupling.** Inside the energy-gradient computation,
   the configuration $q$ and momentum $p$ are concatenated and then
   re-split, so that $p_{\mathrm{split}}$ appears independent of
   $\theta_M$ to the inner autograd. This is required for fp32
   stability (avoids cancellation noise compounded under
   double-backward).
6. **Data-side targets are detached.** $\dot{p}_t^{\mathrm{data}}$ and
   $\widehat{\dot H}_t$ carry no gradient; only the model-side terms
   contribute to backprop.
7. **Optional fixed mass.** When the inverse mass is known
   ($M^{-1}(q) = (1/(m\,l^2))\,I_3$ for a spherical pendulum with
   $m=l=1$), $M_\theta^{-1}$ may be pinned to that constant. This
   removes $M_\theta$'s parameters from the optimiser, eliminates the
   $M$-pretraining stage, and makes $\nabla_q (\tfrac{1}{2} p^T M^{-1} p)$
   identically zero, so $g_{KE} = 0$ and $g_{\mathrm{full}} = g_V$.
