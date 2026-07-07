"""
    StochasticSemiDiscretizationMethod

Moment-stability and stationary-behaviour analysis of linear stochastic delay
differential equations (SDDEs) by semi-discretization of the delayed term.

The workflow is: describe the system as an [`LDDEProblem`](@ref), turn it into a
one-period moment map, and read off its spectral radius (stability) or fixed
point (stationary mean/variance).

- First moment (deterministic): [`DiscreteMapping_M1`](@ref).
- Second moment: the explicit [`DiscreteMapping_M2`](@ref), or the
  memory-lean multiplication-free [`DiscreteMapping_M2_MF`](@ref).
- Solvers: [`spectralRadiusOfMapping`](@ref) / [`fixPointOfMapping`](@ref)
  (explicit), [`spectralRadiusOfMapping_MF`](@ref) /
  [`fixPointOfMapping_MF`](@ref) (``\\mathcal{O}(p^2)`` matrix-free),
  [`spectralRadiusOfMapping_MF_factored`](@ref) /
  [`fixPointOfMapping_MF_factored`](@ref) (Kronecker-factored, high `d`), and
  the CUDA backend [`spectralRadiusOfMapping_GPU`](@ref) /
  [`spectralRadiusOfMapping_auto`](@ref) (loaded via `using CUDA`).

Mean-square stability corresponds to a second-moment spectral radius below `1`.
See the package documentation for worked examples.
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
fixPointOfMapping_GPU

include("precompile.jl")

end # module
