```@meta
CurrentModule = StochasticSemiDiscretizationMethod
```

# StochasticSemiDiscretizationMethod.jl

```@docs
StochasticSemiDiscretizationMethod
```

Moment-stability and stationary-behaviour analysis of linear **stochastic delay
differential equations** (SDDEs) by semi-discretization of the delayed term,
based on Sykora, Bachrathy & Stépán (2019) and the multiplication-free /
Kronecker-factored extensions described in the accompanying paper.

The package handles the general time-periodic linear SDDE (Itô sense)

```math
\mathrm{d}\mathbf{x}(t) = \Big(\mathbf{A}(t)\,\mathbf{x}(t)
  + \sum_{j} \mathbf{B}_j(t)\,\mathbf{x}(t-\tau_j(t)) + \mathbf{c}(t)\Big)\mathrm{d}t
  + \sum_{k}\Big(\boldsymbol{\alpha}^k(t)\,\mathbf{x}(t)
  + \sum_{j}\boldsymbol{\beta}^k_j(t)\,\mathbf{x}(t-\tau_j(t)) + \boldsymbol{\sigma}^k(t)\Big)\mathrm{d}W^k(t),
```

with constant or time-periodic coefficients and constant or time-varying delays,
and returns the **second-moment spectral radius** ``\rho(\mathcal{H})`` (mean-square
stability iff ``\rho(\mathcal{H}) < 1``) and the **stationary covariance**.

## Installation

```julia
using Pkg
Pkg.add("StochasticSemiDiscretizationMethod")
```

The GPU backend is an optional [package extension](@ref gpu); load it with
`using CUDA`.

## Quick start

```julia
using StochasticSemiDiscretizationMethod, StaticArrays

# Damped delay oscillator with delayed multiplicative noise and additive forcing
A = ProportionalMX(@SMatrix [0.0 1.0; -1.0 -0.2])
B = DelayMX(2π, @SMatrix [0.0 0.0; 0.1 0.0])
α = stCoeffMX(1, ProportionalMX(@SMatrix [0.0 0.0;  0.0  0.0]))
β = stCoeffMX(1, DelayMX(2π,  @SMatrix [0.0 0.0; 0.02 0.0]))
σ = stAdditive(1, Additive(@SVector [0.0, 0.5]))
prob = LDDEProblem(A, [B], [α], [β], Additive(2), [σ])

# Multiplication-free second-moment map over one period (τ = 2π, p = 40 steps)
rst = StochasticSemiDiscretizationMethod.calculateResults(
          prob, SemiDiscretization(2, 2π/40), 2π; n_steps = 40, calculate_additive = true)

ρ   = spectralRadiusOfMapping_MF_factored(rst)   # mean-square stable iff ρ < 1
M   = fixPointOfMapping_MF_factored(rst)          # stationary covariance (vech)
var_x = M[1]                                       # stationary variance of x₁
```

## Choosing a method

The quickest way in is the unified interface with a method selector — high-order
Gauss–Legendre collocation by default:

```julia
ρ   = spectralRadiusOfMoment(prob, period, n_steps; method = GaussLegendre(3))  # order 6
var = stationaryVariance(prob, period, n_steps;    method = GaussLegendre(3))
# swap in the classical semi-discretization for a reference/cross-check:
ρsd = spectralRadiusOfMoment(prob, period, n_steps; method = ClassicalSD(2))
```

### Methods and their second-moment convergence order

| Method | Selector | 2nd-moment order | Best for |
|--------|----------|:----------------:|----------|
| Gauss–Legendre collocation | [`Collocation`](@ref)`(S)` / `GaussLegendre(S)` | **2S** (6 at `S=3`) | tight tolerances, low/moderate `d` (**default**) |
| Classical semi-discretization | [`ClassicalSD`](@ref)`(q)` | 1 (any `q`) | reference/cross-check; high `d` via the factored operator |

With `Collocation(S)` the attainable order further depends on the **delay** of
the problem — detected automatically, with one explanatory warning whenever
less than `2S` is attainable (silence with `verbosity = 0`):

| Delay | Engine used | Expected order |
|---|---|:---:|
| constant, grid-aligned (`τ = r·Δt`) | aligned integrated-history | **2S** |
| constant, misaligned | fractional-limit integrated-history | [S+1, 2S] |
| time-periodic smooth `τ(t) ≥ Δt` | fractional-limit integrated-history | floor S+1, measured ≈ 2S |
| varying `τ(t)` with `β ≢ 0`, or multiple delays/channels | classical MF-factored fallback | 1 |

Rough (Wiener-driven) delayed reads — e.g. delayed *velocity* feedback — do
**not** reduce the order of any collocation engine: the delayed drift is kept
exact in pre-integrated history DOFs. Non-smooth **coefficients** do cap the
order for every method (`C⁰` → ≈2, `C¹` → ≈3, jumps → ≈1); see the
[Examples](examples.md) page.

### Lower-level building blocks

| Situation | Function |
|-----------|----------|
| Small step count `p`, reference/debug | [`spectralRadiusOfMapping`](@ref), [`fixPointOfMapping`](@ref) on [`DiscreteMapping_M2`](@ref) |
| Large `p`, moderate state dimension `d` | [`spectralRadiusOfMapping_MF`](@ref), [`fixPointOfMapping_MF`](@ref) |
| Large `p` **and** large `d` (up to hundreds of DOF) | [`spectralRadiusOfMapping_MF_factored`](@ref), [`fixPointOfMapping_MF_factored`](@ref) |
| High order in low/moderate `d` | [`spectralRadiusOfMapping_collocation`](@ref), [`fixPointOfMapping_collocation`](@ref) |
| Very large problems with a CUDA GPU | [`spectralRadiusOfMapping_GPU`](@ref), [`spectralRadiusOfMapping_auto`](@ref) |

See the [Examples](examples.md) and the [API reference](api.md).
