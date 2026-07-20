# High-order Gauss–Legendre collocation solver — user-facing wrappers.
# Adapts an `LDDEProblem` to the internal collocation engines (collocation_engine.jl)
# and exposes second-moment stability / stationary-covariance entry points.
#
# Engine dispatch (automatic, warning-controlled by `verbosity`):
#   constant τ, aligned (τ = r·Δt)  → v9/v8 aligned engine, order 2S
#   constant τ, incommensurate      → vT fractional-limit engine, order [S+1, 2S]
#   function-valued smooth τ(t)     → vT fractional-limit engine, floor S+1
# Scope: a single delay and a single Wiener channel; the vT engine additionally
# requires β ≡ 0 (no delayed multiplicative noise), τ(t) ≥ Δt, T-periodic τ,
# and a uniformly increasing reading map ξ(t) = t − τ(t).

# Build the engine problem from an `LDDEProblem` over one principal period:
# `Prob` for a constant delay, `ProbT` for a function-valued delay.
function _collocation_prob(prob::LDDEProblem, period::Real)
    length(prob.Bs) == 1 ||
        error("the collocation solver supports a single delay term (got $(length(prob.Bs)))")
    (length(prob.αs) ≤ 1 && length(prob.βs) ≤ 1 && length(prob.σs) ≤ 1) ||
        error("the collocation solver supports a single Wiener channel")
    d = size(prob.A(0.0), 1)
    τraw = prob.Bs[1].τ.τ
    A = t -> Matrix{Float64}(prob.A(t))
    B = t -> Matrix{Float64}(prob.Bs[1](t))
    α = isempty(prob.αs) ? (t -> zeros(d, d)) : (t -> Matrix{Float64}(prob.αs[1](t)))
    β = isempty(prob.βs) ? (t -> zeros(d, d)) : (t -> Matrix{Float64}(prob.βs[1](t)))
    σ = isempty(prob.σs) ? (t -> zeros(d, 1)) :
        (t -> reshape(Vector{Float64}(prob.σs[1].V(t)), d, :))
    if τraw isa Real
        return Prob(d, float(period), float(τraw), A, B, α, β, σ)
    end
    τf = t -> float(τraw(t))
    τs = τf.(range(0.0, float(period); length=129))
    all(isfinite, τs) ||
        error("the delay function τ(t) returned a non-finite value on [0, T]")
    ProbT(d, float(period), τf, minimum(τs), maximum(τs), A, B, α, β, σ)
end

# integer lag r if τ = r·(T/p) within tolerance, else 0
function _aligned_r(τ::Real, T::Real, p::Integer)
    h = T / p
    r = round(Int, τ / h)
    (r ≥ 1 && abs(r * h - τ) < 1e-9 * max(τ, 1.0)) ? r : 0
end

# engine-build dispatch (the "one interface" core): pick the best available
# engine for the detected delay class and warn when the attainable order is
# below the 2S the aligned engine advertises.
function _build_collocation(pb::Prob, S::Integer, p::Integer;
                            force::Bool=false, verbosity::Integer=1)
    _aligned_r(pb.τ, pb.T, p) > 0 && return build_v9m(pb, S, p; force=force)
    verbosity ≥ 1 &&
        @warn "constant delay τ = $(pb.τ) is not an integer multiple of Δt = T/n_steps " *
              "= $(pb.T/p): using the fractional-limit engine — order floor $(S+1) " *
              "(observed up to $(2S)) instead of the aligned engine's 2S = $(2S). " *
              "Choose n_steps with τ·n_steps/T integer to restore superconvergence. " *
              "(suppress with verbosity=0)" maxlog=1
    τ0 = pb.τ
    pbT = ProbT(pb.d, pb.T, t -> τ0, τ0, τ0, pb.A, pb.B, pb.α, pb.β, pb.σ)
    build_vT(pbT, S, p; force=force)
end
function _build_collocation(pb::ProbT, S::Integer, p::Integer;
                            force::Bool=false, verbosity::Integer=1)
    verbosity ≥ 1 &&
        @warn "function-valued delay τ(t): using the time-varying-delay collocation " *
              "engine — order floor $(S+1) (observed in [$(S+1), $(2S)]); the 2S = $(2S) " *
              "superconvergence of the constant aligned-delay engine is not attainable " *
              "for a varying delay. (suppress with verbosity=0)" maxlog=1
    build_vT(pb, S, p; force=force)
end

"""
    spectralRadiusOfMapping_collocation(prob::LDDEProblem, period, n_steps; S=3, force=false, verbosity=1, kwargs...) -> Float64

Second-moment spectral radius ``\\rho(\\mathcal{H})`` of the one-period map, built
with an **``S``-stage Gauss–Legendre collocation** discretization of the delayed
term. Mean-square stability corresponds to ``\\rho(\\mathcal{H}) < 1``.

The engine is selected automatically from the delay of `prob`:

  - **constant delay** with ``\\tau = r\\,(T/p)``, integer ``r \\ge 1`` — the aligned
    integrated-history engine, measured order ``2S`` (e.g. `S=3` gives order 6);
  - **constant delay, incommensurate** — the fractional-limit engine, guaranteed
    order floor ``S{+}1``, observed in ``[S{+}1, 2S]`` (a warning suggests the
    aligning `n_steps`);
  - **function-valued smooth delay** ``\\tau(t)`` (`DelayMX` built with a function)
    — the fractional-limit time-varying-delay engine, order floor ``S{+}1``.
    Requires ``\\tau(t) \\ge T/p`` (so `n_steps ≥ ceil(T/min τ)`), a T-periodic
    ``\\tau``, ``\\xi(t)=t-\\tau(t)`` uniformly increasing, and no delayed
    multiplicative noise (``\\beta \\equiv 0``). Delayed reads of **rough**
    (Wiener-driven) components carry no extra order penalty — the delayed drift
    is kept exact in pre-integrated history DOFs, as in the aligned engine.

Compared with the classical semi-discretization solvers
([`spectralRadiusOfMapping_MF`](@ref) and friends, which are first order in the
second moment), the collocation blocks reach high order at a much smaller memory
footprint per accuracy, at the cost of more covariance sub-blocks per delay slot,
which restricts the method to low/moderate state dimension.

`force=true` forces the β-pruned engine even when ``\\beta \\not\\equiv 0`` — the
delayed multiplicative noise is then **ignored** (a warning is emitted); use it
for diagnostics only. `verbosity=0` suppresses the engine-fallback warnings.
Extra `kwargs` (`tol`, `krylovdim`) go to the KrylovKit eigensolver.

Supports a single delay and a single Wiener channel.
"""
function spectralRadiusOfMapping_collocation(prob::LDDEProblem, period::Real,
                                             n_steps::Integer; S::Integer=3,
                                             force::Bool=false, verbosity::Integer=1,
                                             kwargs...)
    eng = _build_collocation(_collocation_prob(prob, period), S, n_steps;
                             force=force, verbosity=verbosity)
    rho_H_krylov_v9m(eng; kwargs...)
end

"""
    fixPointOfMapping_collocation(prob::LDDEProblem, period, n_steps; S=3, force=false, verbosity=1, kwargs...) -> Matrix{Float64}

Stationary second moment (the covariance over the full discretization window,
whose leading ``d\\times d`` block is the state covariance at the period phase) of
a mean-square stable system driven by additive noise, computed with the
``S``-stage Gauss–Legendre collocation discretization. The `[1,1]` entry is the
stationary variance of the first state component.

Requires additive noise (`prob.σs` non-empty) and mean-square stability. See
[`spectralRadiusOfMapping_collocation`](@ref) for the automatic engine
selection (aligned / incommensurate / time-varying delay), the attainable
orders, and the scope restrictions.
"""
function fixPointOfMapping_collocation(prob::LDDEProblem, period::Real,
                                       n_steps::Integer; S::Integer=3,
                                       force::Bool=false, verbosity::Integer=1,
                                       kwargs...)
    eng = _build_collocation(_collocation_prob(prob, period), S, n_steps;
                             force=force, verbosity=verbosity)
    fixPoint_v9m(eng; kwargs...)
end
