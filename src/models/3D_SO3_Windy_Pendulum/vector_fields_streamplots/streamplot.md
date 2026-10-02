# Subnet streamplot — generation steps

Each PDF page is one training-step checkpoint. A page is a $12 \times 3$ grid:
rows = (subnet $\times$ model), cols = body axis $i \in \{x, y, z\}$.

The pipeline up to the field $(\dot\varphi_i,\ \dot p_{\text{subnet},i})$ is **identical to the vector-field plot**. Only the rendering step differs: streamlines instead of arrows, on a denser grid.

## 1. Roll out the predicted trajectory

For each model $\mathcal{M} \in \{\text{GP\_SDE},\ \text{NN\_ODE},\ \text{GT}\}$, simulate from a shared $(R_0, \omega_0)$ and shared noise $\{dW_t\}$:

$$
\{(R_t,\ \omega_t)\}_{t=0}^{T},\qquad q_t = \mathrm{vec}(R_t) \in \mathbb{R}^9.
$$

The trajectory is **not drawn** — used only to anchor the slice.

## 2. Compute predicted momentum

$$
p_t = M(q_t)\,\omega_t = \big[M^{-1}(q_t)\big]^{-1}\,\omega_t.
$$
For GT: $p_t = m\ell^2\,\omega_t$.

## 3. Estimate scale-invariance gauge $\beta$

$$
\beta \;=\; \frac{1/(m\ell^2)}{\frac{1}{T}\sum_t \tfrac{1}{3}\,\mathrm{tr}\!\left(M^{-1}(q_t)\right)}.
$$

Apply $M^{-1}\to \beta\,M^{-1}$, $V\to V/\beta$, $D\to D/\beta$, $B\to B/\beta$.

## 4. Trajectory $\to$ Euler angles (ZYX intrinsic)

$$
\phi = \mathrm{atan2}(R_{21}, R_{22}),\quad
\theta = \mathrm{atan2}(-R_{20},\sqrt{R_{00}^2+R_{10}^2}),\quad
\psi = \mathrm{atan2}(R_{10}, R_{00}).
$$

Yields $\{\varphi_t = (\phi_t,\theta_t,\psi_t)\}$.

## 5. Build the 2D slice for axis $i$

- Anchors:
$$
\bar{\varphi} = \tfrac{1}{T}\sum_t \varphi_t,\qquad \bar{p} = \tfrac{1}{T}\sum_t p_t.
$$
- Regular grid (with $10\%$ padding around the trajectory range):
$$
\varphi^{(k)}_i \in \mathrm{linspace}(\varphi^{\min}_i,\ \varphi^{\max}_i,\ N),\quad
p^{(k)}_i \in \mathrm{linspace}(p^{\min}_i,\ p^{\max}_i,\ N).
$$

A denser grid is used here than in the quiver version (default $N=40$), since `streamplot` interpolates internally and benefits from fine sampling.

## 6. Reconstruct full state at every grid point

Replace only the $i$-th component:
$$
\varphi^{(k)} = \bar{\varphi}\ \text{with}\ [\varphi^{(k)}]_i = \varphi^{(k)}_i,\qquad
p^{(k)} = \bar{p}\ \text{with}\ [p^{(k)}]_i = p^{(k)}_i.
$$

Lift to a rotation matrix:
$$
R^{(k)} = R_z(\psi^{(k)})\,R_y(\theta^{(k)})\,R_x(\phi^{(k)}),\qquad q^{(k)} = \mathrm{vec}(R^{(k)}).
$$

## 7. Evaluate subnetworks at the grid

$$
M^{-1}(q^{(k)}),\quad V(q^{(k)}),\quad D(q^{(k)}),\quad B(q^{(k)}),\quad \nabla_q V(q^{(k)}).
$$
Apply $\beta$.

## 8. Angular velocity

$$
\omega^{(k)} = M^{-1}(q^{(k)})\,p^{(k)}.
$$

## 9. Configuration rate (x-component of the field)

Analytic ZYX Euler-rate map:
$$
\dot\phi   = \omega_x + (\omega_y\sin\phi + \omega_z\cos\phi)\tan\theta,\quad
\dot\theta = \omega_y\cos\phi - \omega_z\sin\phi,\quad
\dot\psi   = \frac{\omega_y\sin\phi + \omega_z\cos\phi}{\cos\theta}.
$$

Pick the $i$-th component $\dot\varphi^{(k)}_i$.

## 10. Subnet $\dot p$ contribution (y-component of the field)

$$
\begin{aligned}
\text{Inverse mass } M^{-1}: &\quad \dot p_M = p \times \omega,\\[2pt]
\text{Potential } V:        &\quad \dot p_V = -\sum_{j=1}^{3} r_j \times \partial_{q_j} V,\\[2pt]
\text{Dissipation } D:      &\quad \dot p_D = -D\,\omega,\\[2pt]
\text{Control } B:          &\quad \dot p_B = B\,u.
\end{aligned}
$$

Take the $i$-th component $[\dot p_{\text{subnet}}^{(k)}]_i$.

## 11. Define the 2D vector field on the grid

For each cell (row = (subnet, model), column = axis $i$):

$$
F^{(k)} \;=\; \big(\,U^{(k)},\ V^{(k)}\,\big) \;=\; \big(\,\dot\varphi^{(k)}_i,\ [\dot p^{(k)}_{\text{subnet}}]_i\,\big).
$$

Speed at each grid point:
$$
s^{(k)} = \sqrt{\big(U^{(k)}\big)^2 + \big(V^{(k)}\big)^2}.
$$

## 12. Render streamlines

`streamplot` integrates the ODE
$$
\frac{d}{d\tau}\begin{pmatrix}\varphi_i\\ p_i\end{pmatrix} = \begin{pmatrix}U(\varphi_i,p_i)\\ V(\varphi_i,p_i)\end{pmatrix}
$$
over the grid (bilinear interpolation between grid points), drawing one streamline through each seed point chosen by the `density` parameter.

Streamline styling:

- **Color** $\propto s^{(k)}$ (model-specific colormap: GP_SDE → Reds, NN_ODE → Blues, GT → Greys).
- **Line width** $\propto 0.5 + 2.0\,s^{(k)}/s_{\max}$.

If $s_{\max} \approx 0$ (e.g. GT $M^{-1}$ row, where $p \times \omega \equiv 0$), the cell is left empty with a small *"field ≡ 0"* annotation.

## Notes

- $\omega$, $\dot\varphi_i$ are $\beta$-invariant; the four $\dot p$-contributions all carry a $1/\beta$ factor.
- A streamline is the integral curve of the subnet field — i.e. the path the system would follow if **only that one subnetwork** drove $\dot p$, with the other three turned off.
- The slice fixes 10 of the 12 phase-space dimensions to $\bar\varphi,\ \bar p$.
