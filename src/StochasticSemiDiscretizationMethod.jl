"""
    StochasticSemiDiscretizationMethod

Moment-stability and stationary-behaviour analysis of linear stochastic delay
differential equations (SDDEs) by discretization of the delayed term into a
one-period moment map.

Despite the historical name, the package is no longer limited to the *classical*
semi-discretization: it also provides a multiplication-free ``\\mathcal{O}(p^2)``
evaluation of the same operator, a Kronecker-factored form for high state
dimension, an optional CUDA backend, and — the current flagship — a **high-order
Gauss–Legendre collocation** solver that reaches order ``2S`` in the second
moment at a much smaller memory footprint per accuracy.

The quickest entry is the unified, method-selecting interface — high-order
collocation by default:

```julia
ρ   = spectralRadiusOfMoment(prob, period, n_steps; method = GaussLegendre(3))  # order 6
var = stationaryVariance(prob, period, n_steps;    method = GaussLegendre(3))
ρsd = spectralRadiusOfMoment(prob, period, n_steps; method = ClassicalSD(2))     # reference
```

selecting between [`Collocation`](@ref)`(S)` (alias `GaussLegendre`, order ``2S``)
and [`ClassicalSD`](@ref)`(q)` (first order, kept for cross-checks). The
lower-level building blocks below are also available directly.

**Time-varying delays.** A smooth, T-periodic, function-valued delay
``\\tau(t) \\ge T/p`` is handled automatically by a fractional-limit
integrated-history collocation engine (guaranteed order floor ``S{+}1``,
measured close to ``2S`` in practice — e.g. spindle-speed-variation turning:
slopes 3.5 at `S=2`, 5.9 at `S=3`); rough (Wiener-driven) delayed reads carry no
order penalty. Problems outside the collocation scope (multiple delays or
channels, delayed multiplicative noise with a varying delay) fall back to the
classical factored path with a warning — see [`Collocation`](@ref).

The general workflow is: describe the system as an [`LDDEProblem`](@ref), turn it
into a one-period moment map, and read off its spectral radius (stability) or
fixed point (stationary mean/variance).

- First moment (deterministic): [`DiscreteMapping_M1`](@ref).
- Second moment: the explicit [`DiscreteMapping_M2`](@ref), or the
  memory-lean multiplication-free [`DiscreteMapping_M2_MF`](@ref).
- Semi-discretization solvers: [`spectralRadiusOfMapping`](@ref) /
  [`fixPointOfMapping`](@ref) (explicit), [`spectralRadiusOfMapping_MF`](@ref) /
  [`fixPointOfMapping_MF`](@ref) (``\\mathcal{O}(p^2)`` matrix-free),
  [`spectralRadiusOfMapping_MF_factored`](@ref) /
  [`fixPointOfMapping_MF_factored`](@ref) (Kronecker-factored, high `d`), and the
  CUDA backend [`spectralRadiusOfMapping_GPU`](@ref) /
  [`spectralRadiusOfMapping_auto`](@ref) (loaded via `using CUDA`).
- High-order collocation (order ``2S``):
  [`spectralRadiusOfMapping_collocation`](@ref) /
  [`fixPointOfMapping_collocation`](@ref).

Mean-square stability corresponds to a second-moment spectral radius below `1`.
The classical/factored path remains the choice for high state dimension and
engineering tolerances; the collocation path wins decisively at tight tolerances
in low/moderate dimension. See the package documentation for worked examples.
"""
module StochasticSemiDiscretizationMethod

import InteractiveUtils
using Reexport
@reexport using LinearAlgebra
@reexport using SparseArrays
@reexport using StaticArrays
@reexport using Arpack
using KrylovKit
using QuadGK
using Lazy: iterated, take

import SemiDiscretizationMethod
import SemiDiscretizationMethod:
AbstractLDDEProblem, AbstractResult,
MatrixOrFunction, ArrayOrFunction, VectorOrFunction, RealOrFunction, CyclicVector,
DiscretizationMethod, SemiDiscretization, NumericSD, methodorder, lagr_el0,
Coefficients, CoefficientMatrix, AdditiveVector,
ProportionalMX, DelayMX, Additive,
Delay,
subArray, SubMX, SubV,
calculate_Aavgs,
# calculateResults,
addSubmatrixToResult!,addSubvectorToResults!,
DiscreteMapping, DiscreteMappingSteps, 
subMxRange, rOfDelay, nStepOfLength, prodl,
reduce_additive

calculateDetResults! = SemiDiscretizationMethod.calculateResults!

include("structures_input.jl")
include("structures_method.jl")
include("structures_result.jl")

include("functions_method.jl")
include("functions_stoch_utilities.jl")
include("functions_discretization.jl")
include("functions_multifree.jl")
include("functions_multifree_factored.jl")
include("functions_gpu_stubs.jl")   # GPU methods live in ext/…CUDAExt.jl (weakdep)
include("collocation_engine.jl")    # internal high-order Gauss–Legendre engine
include("collocation.jl")           # user-facing collocation wrappers
include("moment_methods.jl")        # unified method-selecting interface

export  SemiDiscretization, NumericSD,
ProportionalMX,
Delay,DelayMX,
stCoeffMX,
Additive, stAdditive,
LDDEProblem,
DiscreteMapping_M1, DiscreteMapping_M2,
DiscreteMapping_M2_MF,
MxToCovVec, VecToCovMx,
# DiscreteMapping_M1_1step, DiscreteMapping_M2_1step,
fixPointOfMapping, spectralRadiusOfMapping,
spectralRadiusOfMapping_MF,
fixPointOfMapping_MF,
spectralRadiusOfMapping_MF_factored,
fixPointOfMapping_MF_factored,
spectralRadiusOfMapping_GPU,
spectralRadiusOfMapping_auto,
fixPointOfMapping_GPU,
spectralRadiusOfMapping_collocation,
fixPointOfMapping_collocation,
MomentMethod, Collocation, GaussLegendre, ClassicalSD,
spectralRadiusOfMoment, stationaryVariance, timePeriodicVariance

include("precompile.jl")

end # module
