# Unified moment-stability interface with a selectable discretization method.
# Users typically stay with the high-order Gauss–Legendre collocation
# (`Collocation`/`GaussLegendre`, order 2S); the classical semi-discretization
# (`ClassicalSD`, first order in the second moment) is retained as an explicit
# option, chiefly for cross-checks and regression testing.

"""
    MomentMethod

Abstract supertype of the moment-map discretization selectors passed to
[`spectralRadiusOfMoment`](@ref) and [`stationaryVariance`](@ref):
[`Collocation`](@ref) (alias `GaussLegendre`) and [`ClassicalSD`](@ref).
"""
abstract type MomentMethod end

"""
    Collocation(S=3)
    GaussLegendre(S=3)

High-order ``S``-stage Gauss–Legendre collocation discretization of the delayed
term: **order ``2S``** in the second moment (e.g. `S=3` → order 6). The default
and recommended method; restricted to a single delay and a single Wiener channel.
`GaussLegendre` is an alias.
"""
struct Collocation <: MomentMethod
    S::Int
end
Collocation(; S::Int=3) = Collocation(S)
const GaussLegendre = Collocation

"""
    ClassicalSD(q=2)

Classical semi-discretization with interpolation order `q` of the deterministic
part. It is **first order in the second moment for any `q`** (only the
deterministic mapping benefits from higher `q`); kept as an explicit,
well-understood reference method. Evaluated through the multiplication-free
factored operator, so it also scales to high state dimension.
"""
struct ClassicalSD <: MomentMethod
    q::Int
end
ClassicalSD(; q::Int=2) = ClassicalSD(q)

# largest delay over the period (buffer length for the classical path)
function _period_maxdelay(prob::LDDEProblem, period::Real)
    tmax = 0.0
    for B in prob.Bs
        τ = B.τ.τ
        if τ isa Real
            tmax = max(tmax, float(τ))
        else
            for t in range(0.0, float(period); length=64)
                tmax = max(tmax, float(τ(t)))
            end
        end
    end
    tmax
end

function _classical_result(prob::LDDEProblem, period::Real, n_steps::Integer, q::Int; additive::Bool)
    τmax = _period_maxdelay(prob, period)
    Δt = float(period) / n_steps
    calculateResults(prob, SemiDiscretization(q, Δt), τmax;
                     n_steps=n_steps, calculate_additive=additive)
end

"""
    spectralRadiusOfMoment(prob::LDDEProblem, period, n_steps; method=Collocation(3), kwargs...) -> Float64

Second-moment spectral radius ``\\rho(\\mathcal{H})`` of the one-period map for the
stochastic delay problem `prob`, over the principal `period` resolved with
`n_steps` steps. `method` selects the discretization — [`Collocation`](@ref)`(S)`
(order ``2S``, the default) or [`ClassicalSD`](@ref)`(q)` (first order, reference).
Mean-square stability corresponds to a value below `1`.
"""
spectralRadiusOfMoment(prob::LDDEProblem, period::Real, n_steps::Integer;
                       method::MomentMethod=Collocation(3), kwargs...) =
    _spectral_moment(prob, period, n_steps, method; kwargs...)

_spectral_moment(prob, T, p, m::Collocation; kwargs...) =
    spectralRadiusOfMapping_collocation(prob, T, p; S=m.S, kwargs...)
_spectral_moment(prob, T, p, m::ClassicalSD; kwargs...) =
    spectralRadiusOfMapping_MF_factored(_classical_result(prob, T, p, m.q; additive=false); kwargs...)

"""
    stationaryVariance(prob::LDDEProblem, period, n_steps; method=Collocation(3), kwargs...) -> Float64

Stationary variance of the **first state component** for a mean-square stable,
additively forced system, using the discretization `method` (default
[`Collocation`](@ref)`(3)`, order 6). Requires additive noise (`prob.σs` non-empty)
and stability. For the full stationary covariance use the method-specific
[`fixPointOfMapping_collocation`](@ref) or [`fixPointOfMapping_MF_factored`](@ref).
"""
stationaryVariance(prob::LDDEProblem, period::Real, n_steps::Integer;
                   method::MomentMethod=Collocation(3), kwargs...) =
    _stationary_var(prob, period, n_steps, method; kwargs...)

_stationary_var(prob, T, p, m::Collocation; kwargs...) =
    fixPointOfMapping_collocation(prob, T, p; S=m.S, kwargs...)[1, 1]
_stationary_var(prob, T, p, m::ClassicalSD; kwargs...) =
    fixPointOfMapping_MF_factored(_classical_result(prob, T, p, m.q; additive=true); kwargs...)[1]
