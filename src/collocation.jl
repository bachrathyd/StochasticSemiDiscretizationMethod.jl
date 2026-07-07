# High-order Gauss–Legendre collocation solver — user-facing wrappers.
# Adapts an `LDDEProblem` to the internal collocation engine (collocation_engine.jl)
# and exposes second-moment stability / stationary-covariance entry points.
# Scope of the engine: a single delay τ (constant, with τ = r·Δt, r ≥ 1 integer)
# and a single Wiener channel.

# Build the engine's `Prob` from an `LDDEProblem` over one principal period.
function _collocation_prob(prob::LDDEProblem, period::Real)
    length(prob.Bs) == 1 ||
        error("the collocation solver supports a single delay term (got $(length(prob.Bs)))")
    (length(prob.αs) ≤ 1 && length(prob.βs) ≤ 1 && length(prob.σs) ≤ 1) ||
        error("the collocation solver supports a single Wiener channel")
    d = size(prob.A(0.0), 1)
    τraw = prob.Bs[1].τ.τ
    τraw isa Real ||
        error("the collocation solver requires a constant delay τ = r·Δt (function-valued delays are not supported)")
    A = t -> Matrix{Float64}(prob.A(t))
    B = t -> Matrix{Float64}(prob.Bs[1](t))
    α = isempty(prob.αs) ? (t -> zeros(d, d)) : (t -> Matrix{Float64}(prob.αs[1](t)))
    β = isempty(prob.βs) ? (t -> zeros(d, d)) : (t -> Matrix{Float64}(prob.βs[1](t)))
    σ = isempty(prob.σs) ? (t -> zeros(d, 1)) :
        (t -> reshape(Vector{Float64}(prob.σs[1].V(t)), d, :))
    Prob(d, float(period), float(τraw), A, B, α, β, σ)
end

"""
    spectralRadiusOfMapping_collocation(prob::LDDEProblem, period, n_steps; S=3, force=false, kwargs...) -> Float64

Second-moment spectral radius ``\\rho(\\mathcal{H})`` of the one-period map, built
with an **``S``-stage Gauss–Legendre collocation** discretization of the delayed
term (order ``2S`` in the second moment — e.g. `S=3` gives order 6). Mean-square
stability corresponds to ``\\rho(\\mathcal{H}) < 1``.

Compared with the classical semi-discretization solvers
([`spectralRadiusOfMapping_MF`](@ref) and friends, which are first order in the
second moment), the collocation blocks reach high order at a much smaller memory
footprint per accuracy — at the cost of ``(2S{+}2)^2`` (or ``(S{+}2)^2`` when there
is no delayed multiplicative noise) covariance sub-blocks per delay slot, which
restricts the method to low/moderate state dimension.

`period` is the principal period ``T``; `n_steps` is the number of steps ``p`` per
period; the delay ``\\tau`` is read from `prob` and must satisfy ``\\tau = r\\,(T/p)``
for an integer ``r \\ge 1``. The engine automatically reduces the block size when
there is no delayed multiplicative noise (``\\beta \\equiv 0``); pass `force=true`
to keep the full block. Extra `kwargs` (`tol`, `krylovdim`) go to the KrylovKit
eigensolver.

Supports a single delay and a single Wiener channel.
"""
function spectralRadiusOfMapping_collocation(prob::LDDEProblem, period::Real,
                                             n_steps::Integer; S::Integer=3,
                                             force::Bool=false, kwargs...)
    eng = build_v9m(_collocation_prob(prob, period), S, n_steps; force=force)
    rho_H_krylov_v9m(eng; kwargs...)
end

"""
    fixPointOfMapping_collocation(prob::LDDEProblem, period, n_steps; S=3, force=false, kwargs...) -> Matrix{Float64}

Stationary second moment (the ``d\\times d`` covariance matrix ``\\mathbf{M}^\\ast`` at
the period phase) of a mean-square stable system driven by additive noise,
computed with the ``S``-stage Gauss–Legendre collocation discretization
(order ``2S``). The `[1,1]` entry is the stationary variance of the first state
component.

Requires additive noise (`prob.σs` non-empty) and mean-square stability. See
[`spectralRadiusOfMapping_collocation`](@ref) for the arguments and scope.
"""
function fixPointOfMapping_collocation(prob::LDDEProblem, period::Real,
                                       n_steps::Integer; S::Integer=3,
                                       force::Bool=false, kwargs...)
    eng = build_v9m(_collocation_prob(prob, period), S, n_steps; force=force)
    fixPoint_v9m(eng; kwargs...)
end
