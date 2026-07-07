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

"""
    timePeriodicVariance(prob::LDDEProblem, period, n_steps; component=1, q=2, kwargs...) -> (t, var)

Full **cyclostationary (time-periodic) second-moment solution** over one period,
rather than the single-phase fixed point of [`stationaryVariance`](@ref).

By construction the semi-discretization state carries the history over the buffer
length; padding that buffer to at least one period (`τ_max = max(τ, period)`)
makes the stationary augmented covariance contain the state covariance at every
phase of the period. This returns the variance of `component` sampled at the
`n_steps` phases `t = (0:n_steps-1)·(period/n_steps)` across one period, as
`(t, var)`. `var[1]` equals [`stationaryVariance`](@ref) at the reference phase.

Uses the classical semi-discretization path (order `q`); it is only marginally
more work than the single-phase fixed point (the covariance is already computed —
this just reads its per-lag diagonal blocks). Requires additive noise.
"""
function timePeriodicVariance(prob::LDDEProblem, period::Real, n_steps::Integer;
                              component::Integer=1, q::Integer=2, kwargs...)
    p = n_steps
    τmax = max(_period_maxdelay(prob, period), float(period))   # pad buffer ≥ one period
    Δt = float(period) / p
    rst = calculateResults(prob, SemiDiscretization(q, Δt), τmax;
                           n_steps=p, calculate_additive=true)
    d = size(prob.A(0.0), 1)
    r = div(rst.n, d) - 1
    D = (r + 1) * d
    m = fixPointOfMapping_MF_factored(rst; kwargs...)
    idx = CovVecIdx(D)
    # within-period phase k·Δt ↔ history lag (p−k) mod p (periodicity of the
    # cyclostationary solution); lag 0 is the reference phase.
    var = [ (i = (p - k) % p; m[idx(i*d + component, i*d + component)]) for k in 0:p-1 ]
    t = collect(0:p-1) .* Δt
    (t, var)
end
