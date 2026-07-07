"""
    stCoeffMX(nID::Integer, cMX::CoefficientMatrix)

Stochastic coefficient matrix: wraps a (present `ProportionalMX` or
delayed `DelayMX`) coefficient matrix together with the identifier `nID`
of the independent Wiener process it multiplies. Used to build the
multiplicative-noise terms ``\\boldsymbol{\\alpha}^k(t)`` (present) and
``\\boldsymbol{\\beta}^k(t)`` (delayed) of an [`LDDEProblem`](@ref). Sources
sharing an `nID` are driven by the same Wiener process.
"""
struct stCoeffMX{d,mt<:CoefficientMatrix{d}} <: CoefficientMatrix{d}
    nID::Int64
    cMX::mt
end
(stcm::stCoeffMX)(t) = stcm.cMX(t)

"""
    stAdditive(nID::Integer, V::AdditiveVector)

Stochastic additive source: an `Additive` vector ``\\boldsymbol{\\sigma}^k(t)``
driven by the independent Wiener process identified by `nID`. Each distinct
`nID` is an independent noise channel, so its variance contribution is counted
once. Used for the additive (state-independent) forcing of an
[`LDDEProblem`](@ref); enabling `calculate_additive = true` lets the moment
solvers return the stationary variance driven by these sources.
"""
struct stAdditive{d,T<:AdditiveVector{d}} <: AdditiveVector{d}
    nID::Int64
    V::T
end
(stV::stAdditive)(t) = stV.V(t)

# Linear Delay Differential Equation Problem
"""
    LDDEProblem(A, Bs, αs, βs, c=Additive(size(A,2)), σs=stAdditive[])

Linear stochastic delay differential equation (SDDE) in the It\\^o sense,

```math
\\mathrm{d}\\mathbf{x}(t) = \\Big(\\mathbf{A}(t)\\,\\mathbf{x}(t)
  + \\textstyle\\sum_j \\mathbf{B}_j(t)\\,\\mathbf{x}(t-\\tau_j(t)) + \\mathbf{c}(t)\\Big)\\mathrm{d}t
  + \\textstyle\\sum_k\\Big(\\boldsymbol{\\alpha}^k(t)\\,\\mathbf{x}(t)
  + \\sum_j \\boldsymbol{\\beta}^k_j(t)\\,\\mathbf{x}(t-\\tau_j(t)) + \\boldsymbol{\\sigma}^k(t)\\Big)\\mathrm{d}W^k(t).
```

# Arguments
- `A::ProportionalMX` — present (non-delayed) drift coefficient ``\\mathbf{A}(t)``.
- `Bs::Vector{<:DelayMX}` — delayed drift coefficients ``\\mathbf{B}_j(t)`` (one per delay).
- `αs::Vector{<:stCoeffMX}` — present multiplicative-noise coefficients ``\\boldsymbol{\\alpha}^k``.
- `βs::Vector{<:stCoeffMX}` — delayed multiplicative-noise coefficients ``\\boldsymbol{\\beta}^k_j``.
- `c::Additive` — deterministic additive forcing ``\\mathbf{c}(t)`` (default: zero).
- `σs::Vector{<:stAdditive}` — additive noise sources ``\\boldsymbol{\\sigma}^k`` (default: none).

Each coefficient may be constant or an explicit function of time. Feed the
problem to [`DiscreteMapping_M2`](@ref) / [`DiscreteMapping_M2_MF`](@ref) (or the
factored/GPU solvers) to obtain the second-moment spectral radius and stationary
covariance.
"""
struct LDDEProblem{d,AT <: ProportionalMX{d,<:MatrixOrFunction},BT <: DelayMX{d,<:Any,<:Any}, cT <: Additive{d,<:VectorOrFunction}, αT<:stCoeffMX{d,<:ProportionalMX{d,<:Any}}, βT<:stCoeffMX{d,<:DelayMX{d,<:Any,<:Any}}, σT<:AdditiveVector{d}} <: AbstractLDDEProblem{d}
    A::AT # Coefficient of the proportional term (present /non-discretised term)
    Bs::Vector{BT} # Coefficient of the delayed terms
    αs::Vector{αT} # Noise from the delay matrix (only single delay!!!)
    βs::Vector{βT} # Noise from the delay matrix (only single delay!!!)
    c::cT # Additive vector
    σs::Vector{σT} # Additive vectors for the noises
    w::Int64 # Number of noise sources
end

function LDDEProblem(A::ProportionalMX, Bs::Vector{<:DelayMX}, αs::Vector{<:stCoeffMX}, βs::Vector{<:stCoeffMX}, c::Additive=Additive(size(A, 2)), σs::Vector{<:stAdditive}=Vector{AdditiveVector}(undef,0))
    w::Int64 = all(isempty.((αs,βs,σs))) ? 0 : maximum(αβ.nID for αβ in (αs...,βs...,σs...))
    LDDEProblem(A, Bs, αs, βs, c, σs, w)
end