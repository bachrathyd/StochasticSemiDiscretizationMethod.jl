```@meta
CurrentModule = StochasticSemiDiscretizationMethod
```

# Examples

Runnable versions of these (and more, including stability charts produced with
[MDBM.jl](https://github.com/bachrathyd/MDBM.jl)) live in the `examples/` folder
of the repository. Set that folder up once with

```julia
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

## Scalar Hayes equation

The stochastic Hayes equation ``\mathrm{d}x = (a\,x + b\,x(t-1))\,\mathrm{d}t
+ \beta\,x(t-1)\,\mathrm{d}W`` has a one-dimensional state and delayed
multiplicative noise.

```julia
using StochasticSemiDiscretizationMethod

a, b, β = -6.0, 2.0, 0.5
A  = ProportionalMX(fill(a, 1, 1))
B  = DelayMX(1.0, fill(b, 1, 1))
α  = stCoeffMX(1, ProportionalMX(zeros(1, 1)))
βM = stCoeffMX(1, DelayMX(1.0, fill(β, 1, 1)))
σ  = stAdditive(1, Additive(ones(1)))
prob = LDDEProblem(A, [B], [α], [βM], Additive(1), [σ])

rst = StochasticSemiDiscretizationMethod.calculateResults(
          prob, SemiDiscretization(0, 0.1), 1.0; n_steps = 10, calculate_additive = true)

ρ     = spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst))
var_x = fixPointOfMapping_MF(DiscreteMapping_M2_MF(rst))[1]
```

## Stochastic delayed Mathieu oscillator

A time-periodic, second-order oscillator with both a periodic delayed term and
present/delayed multiplicative noise — the representative case where *every*
per-step coefficient differs.

```julia
using StochasticSemiDiscretizationMethod, StaticArrays

A(t) = @SMatrix [0.0 1.0; -(3.0 + 2.0*cos(t)) -0.2]
B(t) = @SMatrix [0.0 0.0; 0.5*(1 + 0.4cos(t)) 0.0]
a(t) = @SMatrix [0.0 0.0; -0.1*(3.0 + 2.0*cos(t)) -0.02]
b(t) = @SMatrix [0.0 0.0; 0.05*(1 + 0.4cos(t)) 0.0]

prob = LDDEProblem(
    ProportionalMX(A), [DelayMX(2π, B)],
    [stCoeffMX(1, ProportionalMX(a))],
    [stCoeffMX(1, DelayMX(2π, b))],
    Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.1]))])

# principal period P = 2π resolved with p = 40 steps
rst = StochasticSemiDiscretizationMethod.calculateResults(
          prob, SemiDiscretization(2, 2π/40), 2π; n_steps = 40, calculate_additive = true)

ρ     = spectralRadiusOfMapping_MF_factored(rst)
var_x = fixPointOfMapping_MF_factored(rst)[1]
```

## High-order collocation (recommended for tight tolerances)

The Gauss–Legendre collocation solver reaches order ``2S`` in the second moment.
`S = 3` (order 6) is a good default. It takes the principal period `T` and the
number of steps `p` per period; a constant delay aligned with the grid
(``\tau = r\,(T/p)``, integer `r`) gets the full ``2S``, and other delay classes
are dispatched automatically — see the sections below.

```julia
using StochasticSemiDiscretizationMethod, StaticArrays

# delayed-PD-drift stochastic Mathieu oscillator (τ = T = 1), additive noise
A(t) = @SMatrix [0.0 1.0; -(1.0 + 0.5cos(2π*t)) -0.4]
B(t) = @SMatrix [0.0 0.0; 0.20*(1 + 0.3cos(2π*t)) 0.12*(1 + 0.4cos(2π*t))]
α(t) = @SMatrix [0.0 0.0; 0.30 0.0]
β(t) = @SMatrix [0.0 0.0; 0.0  0.0]
prob = LDDEProblem(ProportionalMX(A), [DelayMX(1.0, B)],
    [stCoeffMX(1, ProportionalMX(α))], [stCoeffMX(1, DelayMX(1.0, β))],
    Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])

T = 1.0
ρ     = spectralRadiusOfMapping_collocation(prob, T, 12; S = 3)   # order 6 in ~12 steps
var_x = fixPointOfMapping_collocation(prob, T, 12; S = 3)[1, 1]
```

A handful of steps at `S = 3` already matches what the classical scheme needs
hundreds of steps to reach.

## Delay classes — defining each variant properly

The collocation interface detects the delay class from how the system is
*defined*; nothing else changes. The mini-demos below share this prelude
(1-DOF oscillator, `x = [position; velocity]`):

```julia
using StochasticSemiDiscretizationMethod, StaticArrays
A(t) = @SMatrix [0.0 1.0; -(1.0 + 0.5cos(2π*t)) -0.4]
α(t) = @SMatrix [0.0 0.0; 0.25 0.0]
z2   = @SMatrix zeros(2, 2)
σ    = @SVector [0.0, 0.3]
T = 1.0; p = 16
mkprob(τ, B; β = t -> z2) =
    LDDEProblem(ProportionalMX(A), [DelayMX(τ, B)],
                [stCoeffMX(1, ProportionalMX(α))], [stCoeffMX(1, DelayMX(τ, β))],
                Additive(2), [stAdditive(1, Additive(σ))])
```

### Constant aligned delay, smooth read (order 2S)

Delayed **position** feedback; `τ = 0.5 = 8·Δt` lands on the grid:

```julia
B(t) = @SMatrix [0.0 0.0; 0.2 0.0]          # reads x₁ (position) — smooth
ρ = spectralRadiusOfMoment(mkprob(0.5, B), T, p)
```

### Constant aligned delay, rough read (still order 2S)

Delayed **velocity** feedback (the delayed-D term of a PD controller): the read
component is Wiener-driven ("rough"), but the integrated-history blocks keep
the full order:

```julia
Bpd(t) = @SMatrix [0.0 0.0; 0.2 0.12]       # also reads x₂ (velocity) — rough
ρ = spectralRadiusOfMoment(mkprob(0.5, Bpd), T, p)
```

### Constant misaligned delay (order in [S+1, 2S])

`τ ≠ r·Δt` routes to the fractional-limit engine; the warning suggests the
aligning `n_steps` — often the better fix:

```julia
ρ = spectralRadiusOfMoment(mkprob(0.618, B), T, p)                 # warns
ρ = spectralRadiusOfMoment(mkprob(0.618, B), T, p; verbosity = 0)  # silent
```

### Time-periodic smooth delay (floor S+1, measured ≈ 2S)

Pass the delay as a **function** — that is the entire difference. Requirements:
`τ(t) ≥ Δt` (an error reports the minimum `n_steps`), `τ` T-periodic and
smooth, `ξ(t) = t − τ(t)` increasing — a **one-sided** bound `τ′(t) ≤ 0.9`
(the delay may *decrease* arbitrarily fast):

```julia
τfun(t) = 0.45 + 0.08sin(2π*t)              # e.g. spindle-speed variation
ρ   = spectralRadiusOfMoment(mkprob(τfun, B), T, p)
var = stationaryVariance(mkprob(τfun, B), T, p)
```

On the SSV turning model the measured orders are 3.5 at `S = 2` and 5.9 at
`S = 3` — near the superconvergent `2S`, well above the guaranteed `S+1` floor.
Rough reads (`Bpd`) work identically.

### Delayed multiplicative noise (β ≢ 0)

Supported for **every** delay class. The block additionally carries point-sample
DOFs at the delayed reading positions, filled from the same causal two-time
kernel, so the order matches the drift-only case (aligned `2S`, varying floor
`S+1`) and **rough** delayed-noise reads carry no penalty:

```julia
βn(t) = @SMatrix [0.0 0.0; 0.1 0.0]         # noise reads the delayed state
ρ = spectralRadiusOfMoment(mkprob(0.5,  B; β = βn), T, p)  # aligned → order 2S
ρ = spectralRadiusOfMoment(mkprob(τfun, B; β = βn), T, p)  # varying → floor S+1
```

Only multiple delays or Wiener channels fall back to the classical path.

### Non-smooth coefficients can cap the order — for every method

The orders above assume smooth coefficients `A(t), B(t), α(t), σ(t)`. A `C⁰`
coefficient (derivative kinks, e.g. the trapezoidal engagement of a helical
milling cutter) caps the attainable order at ≈2; `C¹` at ≈3; genuine jumps
(straight-fluted milling) at ≈1 — regardless of `S` or `q`. If the kink times
are fixed, choose `n_steps` so they land on step boundaries (full order
returns); under spindle-speed variation they drift, so expect the cap
asymptotically — though at engineering tolerances the high-order engine usually
still wins on error constants. A smoothed coefficient model (e.g. truncated
Fourier) restores the full order.

## Choosing the CPU or GPU backend automatically

```julia
using StochasticSemiDiscretizationMethod
using CUDA   # loads the GPU extension; without it, auto uses the CPU path

dm = DiscreteMapping_M2_MF(rst)
ρ  = spectralRadiusOfMapping_auto(dm)   # GPU above the crossover, CPU below
```
