# Work-precision demonstration suite — stochastic delay differential equations

This folder demonstrates the **second-moment stability** computation (spectral radius
`ρ(H)` of the second-moment mapping) for a broad catalogue of linear **stochastic delay
differential equations (SDDEs)**, using the trusted **Stochastic Semi-Discretization
Method (SDM)** implemented in `StochasticSemiDiscretizationMethod.jl`.

For every example we produce a **work-precision diagram**: the error of `ρ(H)` versus
computational cost and versus the period resolution `p`, at three SDM accuracy orders.

## What is shown

Each example yields three files:

| file | content |
|------|---------|
| `<name>.csv`        | raw data: `q, order, p, rho, cputime, abserr` + `rho_ref` |
| `<name>.png`        | work-precision: `|ρ(H) − ρ_ref|` vs **CPU time** (log–log) |
| `<name>_order.png`  | convergence order: `|ρ(H) − ρ_ref|` vs **p** (log–log) |

Three curves per plot, the **Lagrange-interpolation order `q` of the delay term**:

* **q = 0** → first order, `O(h¹)`
* **q = 2** → second order, `O(h²)`
* **q = 4** → third order, `O(h³)`

This is the convergence ladder established by Sykora & Bachrathy (Appl. Math. Modelling
88, 2020): increasing the delay-interpolation order raises the convergence order of the
second-moment spectral radius, up to the `O(h³)` ceiling set by the stochastic Itô term.
The reference `ρ_ref` is Richardson-extrapolated from fine `q = 4` runs.

## Coverage (the requested variety)

The 24 examples deliberately span:

* **noise type** — additive, multiplicative (present-state α and/or delayed β), and mixed;
* **period vs delay** — both `T > τ` and `T < τ`;
* **delays** — single, multiple, and **time-varying** `τ(t)`;
* **coefficients** — constant and **periodic** (parametric excitation);
* **dimension** — `d = 1 … 4`;
* **domains** — mechanical / vibration / manufacturing / control engineering, biology,
  ecology, neuroscience, traffic, physics, economics.

## Example catalogue

| # | name | domain | d | noise | notes |
|---|------|--------|---|-------|-------|
| 01 | hayes_scalar | benchmark | 1 | delayed mult. | analytic benchmark, T=τ |
| 02 | hayes_present_delay_noise | benchmark | 1 | present+delayed mult. | T=τ |
| 03 | hayes_additive | benchmark | 1 | additive | drift-governed ρ |
| 04 | delayed_logistic | biology | 1 | mult. | Hutchinson population eqn |
| 05 | mathieu_TgtTau | engineering/parametric | 2 | mult. | **T > τ** (P=4π, τ=2π) |
| 06 | mathieu_TltTau | engineering/parametric | 2 | mult. | **T < τ** (τ=4π, P=2π) |
| 07 | delayed_oscillator_additive | vibration | 2 | additive | ẍ+cẋ+kx=−x(t−τ)+noise |
| 08 | delayed_oscillator_mult | vibration | 2 | mult. (velocity) | |
| 09 | machining_chatter | manufacturing | 2 | mult. | regenerative turning chatter |
| 10 | two_delay_scalar | multi-delay | 1 | mult. | τ₁=0.5, τ₂=1.0 |
| 11 | time_varying_delay | time-varying delay | 1 | mult. | τ(t)=1+0.3 sin t |
| 12 | neural_delayed_feedback | neuroscience | 1 | mixed | Mackey–Glass linearization |
| 13 | gene_regulatory | systems biology | 2 | mixed | delayed repression |
| 14 | predator_prey_delay | ecology | 2 | additive | gestation delay |
| 15 | traffic_carfollowing | traffic | 2 | mixed | driver reaction delay |
| 16 | feedback_control | control | 2 | mixed | actuation delay (can be unstable) |
| 17 | mathieu_strong_excitation | parametric | 2 | mult. | strong ε (often unstable) |
| 18 | chain_3dof | higher-dim | 4 | additive | coupled delayed oscillators |
| 19 | periodic_coeff_scalar | periodic | 1 | additive | periodic drift + noise |
| 20 | inverted_pendulum_delay | control | 2 | mixed | delayed stabilization |
| 21 | two_delay_oscillator | multi-delay | 2 | mult. | τ₁=1, τ₂=2 |
| 22 | delayed_bistable_physics | physics | 1 | mult. | near-threshold, strong noise |
| 23 | tvdelay_periodic_combined | combined | 2 | mult. | periodic coeff + τ(t) |
| 24 | kalecki_business_cycle | economics | 2 | mixed | investment delay |

> Some control / strong-excitation examples (16, 17, 20, 24) are **mean-square unstable**
> (`ρ_ref > 1`) at the chosen parameters — this is physically expected and does not affect
> the validity of the convergence study (the *order* of `ρ(H)` is what the diagrams show).

## How to regenerate

```
julia --project=. demonstration/wp_all.jl all      # all 24
julia --project=. demonstration/wp_all.jl 5        # just example #5
```

Definitions live in `examples.jl` (the `SDDEProblem` catalogue) and `wp_all.jl`
(the driver: SDM via the package, Richardson reference, CSV + PNG output).

## Method note (important)

The diagrams use the **trusted SDM**, whose second-moment machinery is validated: with
noise off it returns exactly `ρ(Φ_det)²`, and it reproduces analytic benchmarks. Its
practical accuracy ceiling for the stochastic second moment is `O(h³)` (the Itô term).

A separate research line (see `../HIGHORDER_M2_PLAN.md`) investigated **collocation of the
deterministic moment equation** to exceed this ceiling. The convergence *order* can be
raised, but the reduced `(M,P)` moment-DDE lift used in the prototype converged to the
**wrong value** (it fails the noise-off `ρ(Φ)²` test), so it is **not** used here. The
correct high-order route — feeding the SDM's own exact-Itô machinery with collocation-order
within-step covariance — is documented as future work in the plan file.
