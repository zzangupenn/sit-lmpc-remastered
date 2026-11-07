# Controller architecture and dynamics learning

## Main findings

- The known-dynamics controller and BNN controller share the MPPI/LMPPI optimization, convex-hull safe-set selection, value model and result formats.
- A Gaussian BNN coefficient is sampled once for a complete dynamics function. That coefficient stays fixed over the horizon and across candidate controls within the same scenario set.
- Online belief learning updates Gaussian coefficients after attempts. Offline population training learns neural features and the prior; the two operations are explicit and separate.
- Dynamics-based chance constraints are independent of learned risk-to-go. The supplied presets disable risk-to-go and alternative safe-set representations.

## One control iteration

1. Read the measured state and shift the previous optimized action horizon.
2. Sample action candidates. With BNN dynamics, draw coherent function scenarios from the currently published Gaussian belief.
3. Predict candidate trajectories and their mean; transform them to track/Frenet coordinates. Query the learned value and the selected convex-hull safe set.
4. Construct active lambda combinations. When chance constraints are disabled, their lambda dimensions and scenario-verification work are absent.
5. With chance constraints enabled, compute boundary and terminal-hull violation probabilities. Prefer a feasible candidate with the restored smaller-lambda ordering. If none is feasible, `fallback: baseline` uses the conservative SIT fallback and logs the selected probabilities; `fallback: stop` ends the attempt without applying a control.
6. Apply the first optimized action to the simulator. Store the measured transition, including the actual applied physical control. Saved snapshots retain the predictor, posterior and experiment-local trajectories needed for resume.

## Gaussian BNN model and posterior

Let $x_t$ be the seven-component ST state, $u_t$ the two-component physical control and $\tilde y_t$ the normalized dynamics target defined by the model's domain transform. The learned model is

$$
\tilde y_t = b_\theta(x_t,u_t) + B_\theta(x_t,u_t)z + \epsilon_t,
\qquad z\sim\mathcal N(m_0,S_0),\quad \epsilon_t\sim\mathcal N(0,R).
$$

Here $b_\theta$ is the neural mean-target map and $B_\theta$ is the neural basis map, whose columns multiply the low-dimensional Gaussian coefficient $z$. Offline population training learns $\theta$, the Gaussian prior parameters and observation-noise parameters. The domain transform converts the predicted target back to a next ST state; residual checkpoints include their configured analytical baseline.

For fitting transition $i$, define $B_i=B_\theta(x_i,u_i)$ and $r_i=\tilde y_i-b_\theta(x_i,u_i)$. With calibrated evidence power $\tau$, the Gaussian update is

$$
S_\tau^{-1}=S_0^{-1}+\tau\sum_i B_i^\top R^{-1}B_i,
\qquad
m_\tau=S_\tau\left(S_0^{-1}m_0+\tau\sum_iB_i^\top R^{-1}r_i\right).
$$

Each unique fitting transition contributes once. Successful and failed attempts supply measured data; disjoint calibration/test windows are excluded from fitting. Tempering selects an evidence power against rollout coverage and energy-score criteria, and can retain the previous published belief when a candidate fails calibration. It does not retrain neural features or identify a literal friction value.

For scenario $j$, draw $z^{(j)}=m_\tau+L_\tau\xi^{(j)}$, where $L_\tau L_\tau^\top=S_\tau$ and $\xi^{(j)}\sim\mathcal N(0,I)$. Reuse $z^{(j)}$ for all transitions of that scenario. Scenarios sampled directly from the Gaussian are equally weighted; multiplying by their Gaussian density again would be incorrect. Nominal predictions average the coherent state trajectories, with circular handling of angles. No extra physical process noise is injected into these function samples.

## Chance constraints

For $M$ sampled functions, the horizon boundary estimate for candidate controls $U$ is

$$
\widehat p_{\mathrm{boundary}}(U)
=\frac1M\sum_{j=1}^M\mathbf1\{\exists k:\ |e_y(x_k^{(j)})|>w\},
$$

where $e_y$ is lateral track error and $w$ is the configured inward boundary limit. Terminal-hull violation is

$$
\widehat p_{\mathrm{hull}}(U)
=\frac1M\sum_{j=1}^M\mathbf1\{d_{\mathcal S}(x_H^{(j)})>\delta\}.
$$

The safe-set hull $\mathcal S$, distance $d_{\mathcal S}$ and relaxation $\delta$ use the same local geometry as the nominal controller. Nonfinite/invalid scenarios count as violations. The constraints compare these probabilities with `boundary_threshold` and `terminal_hull_threshold`.

`chance_estimation: planning_samples` reuses the planning functions and reports empirical fractions. `independent` draws separate verification functions and uses the implemented simultaneous Monte Carlo upper bound. Neither mode certifies the true plant when the learned distribution is misspecified. The BNN preset retains the existing v2 experiment's permissive terminal-hull threshold of 0.9; a smaller value requires more sampled functions to satisfy the hull.

**Table 1. Configuration boundaries.**

| Setting | Effect |
|---|---|
| `planner.sit_lmpc_mode: true` | Fixed analytical predictor; stop on first physical failure |
| `planner.dynamics.backend: gaussian_bnn` | Learned population predictor; requires SIT mode and finite-bank learning disabled |
| `planner.dynamics.belief_update: posterior` | Attempt-boundary Gaussian latent updates |
| `planner.dynamics.posterior_method: tempered` | Calibrated evidence-power selection |
| `planner.probabilistic_safety.enabled` | Dynamics-based chance constraints and their lambda sampling |
| `value.model` | NF or MLP value learning, independent of dynamics choice |
| `diagnostics.keyboard_controls` | Keyboard debug plots, independent of simulator rendering |

## Source boundaries

The implementation is a renamed racecar extraction from the recorded BRO-LMPC working files, not a new numerical rewrite. Auxiliary internal modules remain to preserve runner/checkpoint compatibility; the default presets select only the requested baseline or BNN/chance path. `vendor/dyna-dist`, `vendor/f1tenth_gym` and `vendor/utilitySuite` are source snapshots with original notices. Changes made here do not alter their original checkouts.

## Validated example

The [racecar example](../examples/racecar/README.md) includes both 300-lap controller configurations and measured results. It is the portable entry point for reproducing the known-dynamics and BNN/chance paths.
