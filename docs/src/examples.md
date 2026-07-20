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
number of steps `p` per period (with the delay ``\tau = r\,(T/p)`` for integer
`r`).

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

## Choosing the CPU or GPU backend automatically

```julia
using StochasticSemiDiscretizationMethod
using CUDA   # loads the GPU extension; without it, auto uses the CPU path

dm = DiscreteMapping_M2_MF(rst)
ρ  = spectralRadiusOfMapping_auto(dm)   # GPU above the crossover, CPU below
```
