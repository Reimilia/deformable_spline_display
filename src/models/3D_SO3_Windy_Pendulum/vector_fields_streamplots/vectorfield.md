# Subnet vector-field plots — generation steps

Each PDF page is one training-step checkpoint. A page is a $12 \times 3$ grid:
rows = (subnet $\times$ model), cols = body axis $i \in \{x, y, z\}$.

## 1. Roll out the predicted trajectory

For each model $\mathcal{M} \in \{\text{GP\_SDE},\ \text{NN\_ODE},\ \text{GT}\}$, simulate the dynamics from a shared initial condition $(R_0, \omega_0)$ and shared noise sequence $\{dW_t\}$ over $T$ steps:

$$
\{(R_t,\ \omega_t)\}_{t=0}^{T},\qquad q_t = \mathrm{vec}(R_t) \in \mathbb{R}^9,\ \omega_t \in \mathbb{R}^3.
$$

The trajectory is **not drawn**. It is used only as an anchor.

## 2. Compute predicted momentum along the trajectory

$$
p_t \;=\; \frac{1}{\beta}\,M(q_t)\,\omega_t \;=\; \big[\beta\,M^{-1}(q_t)\big]^{-1}\,\omega_t.
$$

For GT, $\beta = 1$ and $M^{-1}_\text{GT} = \tfrac{1}{m\ell^2} I_3$, so $p_t = m\ell^2\,\omega_t$.

(The scalar $\beta$ is the gauge from step 3 — defined below. Under the PHS gauge $(M, p, V, D, B) \to \alpha\,(M, p, V, D, B)$ with $\alpha = 1/\beta$, dividing $p$ by $\beta$ keeps $\omega = M^{-1}p$ invariant when $M^{-1}$ is multiplied by $\beta$.)

## 3. Estimate the scale-invariance gauge $\beta$

$$
\beta \;=\; \frac{1/(m\ell^2)}{\frac{1}{T}\sum_t \frac{1}{3}\,\mathrm{tr}\!\left(M^{-1}(q_t)\right)}.
$$

Apply
$$
M^{-1}\to \beta\, M^{-1},\quad V\to V/\beta,\quad D\to D/\beta,\quad B\to B/\beta,
$$
which makes the model output match the GT scale.

## 4. Convert the trajectory to Euler angles

ZYX intrinsic Euler angles $\varphi_t = (\phi_t,\ \theta_t,\ \psi_t)$ are extracted from $R_t$:
$$
\phi = \mathrm{atan2}(R_{21}, R_{22}),\quad
\theta = \mathrm{atan2}(-R_{20},\sqrt{R_{00}^2+R_{10}^2}),\quad
\psi = \mathrm{atan2}(R_{10}, R_{00}).
$$

## 5. Pick a 2D slice for axis $i$

For each body axis $i \in \{x, y, z\}$:

- Slice anchors (means over the predicted trajectory):
$$
\bar{\varphi} = \frac{1}{T}\sum_t \varphi_t,\qquad \bar{p} = \frac{1}{T}\sum_t p_t.
$$
- Grid extent (with $10\%$ padding) along the $i$-th components:
$$
[\varphi^{\min}_i,\ \varphi^{\max}_i],\qquad [p^{\min}_i,\ p^{\max}_i].
$$
- Build a regular $N\times N$ grid:
$$
\varphi^{(k)}_i \in \mathrm{linspace}(\varphi^{\min}_i,\ \varphi^{\max}_i,\ N),\qquad
p^{(k)}_i \in \mathrm{linspace}(p^{\min}_i,\ p^{\max}_i,\ N).
$$

## 6. Reconstruct the full state at every grid point

For each grid point $(\varphi^{(k)}_i,\ p^{(k)}_i)$ replace only the $i$-th component:

$$
\varphi^{(k)} = \bar{\varphi} \text{ with } [\varphi^{(k)}]_i = \varphi^{(k)}_i,\qquad
p^{(k)} = \bar{p} \text{ with } [p^{(k)}]_i = p^{(k)}_i.
$$

Convert Euler angles to a rotation matrix:
$$
R^{(k)} \;=\; R_z(\psi^{(k)})\,R_y(\theta^{(k)})\,R_x(\phi^{(k)}),\qquad q^{(k)} = \mathrm{vec}(R^{(k)}).
$$

## 7. Evaluate the model subnetworks at every grid point

$$
M^{-1}(q^{(k)}),\quad V(q^{(k)}),\quad D(q^{(k)}),\quad B(q^{(k)}),
$$
and the potential gradient via autograd:
$$
\nabla_q V(q^{(k)}) \in \mathbb{R}^9.
$$
Apply $\beta$ from step 3.

## 8. Compute angular velocity at every grid point

$$
\omega^{(k)} \;=\; M^{-1}(q^{(k)})\,p^{(k)} \;\in\; \mathbb{R}^3.
$$

## 9. Compute the arrow x-component (configuration rate)

Analytic ZYX Euler-rate map:
$$
\dot{\phi}   = \omega_x + (\omega_y\sin\phi + \omega_z\cos\phi)\tan\theta,\quad
\dot{\theta} = \omega_y\cos\phi - \omega_z\sin\phi,\quad
\dot{\psi}   = \frac{\omega_y\sin\phi + \omega_z\cos\phi}{\cos\theta}.
$$

Take $\dot{\varphi}^{(k)}_i$ — the $i$-th component.

## 10. Compute the arrow y-component (subnet $\dot p$ contribution)

For the subnet of the row, evaluate the contribution to $\dot p \in \mathbb{R}^3$ from the PHS spec:

$$
\begin{aligned}
\text{Inverse mass } M^{-1}: &\quad \dot p_M = p \times \omega,\\[2pt]
\text{Potential } V:        &\quad \dot p_V = -\,\sum_{j=1}^{3} r_j \times \partial_{q_j} V,\\[2pt]
\text{Dissipation } D:      &\quad \dot p_D = -\,D\,\omega,\\[2pt]
\text{Control } B:          &\quad \dot p_B = B\,u.
\end{aligned}
$$

Take the $i$-th component $[\dot p_{\text{subnet}}^{(k)}]_i$.

## 11. Draw the quiver

For the cell at row (subnet, model) and column $i$, plot one arrow per grid point:

$$
\text{arrow tail} = \big(\varphi^{(k)}_i,\ p^{(k)}_i\big),\qquad
\text{arrow vector} = \big(\dot{\varphi}^{(k)}_i,\ [\dot p^{(k)}_{\text{subnet}}]_i\big).
$$

The arrow shows where **only that subnetwork** would push the system in the $(\varphi_i, p_i)$ phase plane.

## Notes

- $\omega$, $\dot{\varphi}_i$ are $\beta$-invariant. The four $\dot p$-contributions all scale by $1/\beta$.
- For GT, $M^{-1} \propto I$, hence $p \times \omega \equiv 0$ — the $M^{-1}$ row arrows have zero $y$-component.
- The slice fixes 10 of the 12 phase-space dimensions to $\bar{\varphi}, \bar{p}$, so the field shown is a 2D restriction of the full 12-D dynamics.
