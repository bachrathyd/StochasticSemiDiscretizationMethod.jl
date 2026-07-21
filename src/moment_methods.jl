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
term. The attainable order depends on the delay of the problem (detected
automatically; a warning names the engine used unless `verbosity=0`):

  - constant delay aligned with the grid (``\\tau = r\\,T/p``): **order ``2S``**
    (e.g. `S=3` → order 6);
  - constant incommensurate delay: order in ``[S{+}1, 2S]``;
  - smooth, T-periodic, function-valued delay ``\\tau(t) \\ge T/p``: **order floor
    ``S{+}1``**, observed in ``[S{+}1, 2S]`` (requires ``\\xi(t) = t - \\tau(t)``
    uniformly increasing). Rough (Wiener-driven) delayed reads carry no extra
    penalty — the delayed drift stays exact in pre-integrated history DOFs.

Delayed **multiplicative** noise (``\\beta \\not\\equiv 0``) is supported for
every delay class: the block additionally carries point-sample DOFs at the
delayed reading positions, filled from the same causal kernel, so rough delayed
*noise* reads keep the order too (aligned ``2S``, varying floor ``S{+}1``).
**Multiple delays** ``\\sum_j \\mathbf{B}_j\\,\\mathbf{x}(t-\\tau_j(t))`` are also
handled — each delay carries its own integrated-history DOFs and shares the
point-sample states; the ``\\beta_j`` pair with the ``\\mathbf{B}_j`` by index.

The default and recommended method; restricted to a **single Wiener channel**.
Multiple independent Wiener channels automatically fall back to
[`ClassicalSD`](@ref)`(2)` with a warning. `GaussLegendre` is an alias.
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

# Collocation applicability: returns `nothing` when the collocation engines can
# handle `prob`, else a human-readable reason for the classical fallback. Multiple
# delays and delayed multiplicative noise are handled by the vT engine; only
# multiple Wiener channels (independent noise sources) still fall back.
function _collocation_blocked(prob::LDDEProblem, period::Real, n_steps::Integer)
    nids = Int[]
    for x in prob.αs; push!(nids, x.nID); end
    for x in prob.βs; push!(nids, x.nID); end
    for x in prob.σs; push!(nids, x.nID); end
    length(unique(nids)) ≤ 1 || return "multiple Wiener channels ($(sort(unique(nids))))"
    nothing
end

# fallback: Collocation requested but not applicable → classical path. NOTE:
# collocation-specific kwargs (tol, krylovdim, force) are NOT forwarded — the
# classical solvers have different keyword surfaces, and the user tuned them
# for the collocation path anyway.
function _collocation_or_fallback(f_colloc, f_classical, prob, T, p, m::Collocation,
                                  verbosity::Integer)
    blocked = _collocation_blocked(prob, T, p)
    blocked === nothing && return f_colloc()
    verbosity ≥ 1 &&
        @warn "Collocation($(m.S)) (order up to $(2m.S)) is not applicable to this " *
              "problem: $blocked. Falling back to the classical multiplication-free " *
              "factored semi-discretization (ClassicalSD(2), first order in the " *
              "second moment); collocation-specific keyword arguments are ignored. " *
              "(suppress with verbosity=0)" maxlog=1
    f_classical()
end

"""
    spectralRadiusOfMoment(prob::LDDEProblem, period, n_steps; method=Collocation(3), verbosity=1, kwargs...) -> Float64

Second-moment spectral radius ``\\rho(\\mathcal{H})`` of the one-period map for the
stochastic delay problem `prob`, over the principal `period` resolved with
`n_steps` steps. `method` selects the discretization — [`Collocation`](@ref)`(S)`
(the default; order ``2S`` for an aligned constant delay, floor ``S{+}1`` for a
smooth time-varying delay — see [`Collocation`](@ref)) or [`ClassicalSD`](@ref)`(q)`
(first order, reference). The problem class is detected automatically and the
best available engine is used; whenever the attainable order is below what the
requested method advertises (or a fallback is taken), one warning explains the
choice — set `verbosity=0` to silence. Mean-square stability corresponds to a
value below `1`.
"""
spectralRadiusOfMoment(prob::LDDEProblem, period::Real, n_steps::Integer;
                       method::MomentMethod=Collocation(3), kwargs...) =
    _spectral_moment(prob, period, n_steps, method; kwargs...)

_spectral_moment(prob, T, p, m::Collocation; verbosity::Integer=1, kwargs...) =
    _collocation_or_fallback(
        () -> spectralRadiusOfMapping_collocation(prob, T, p; S=m.S,
                                                  verbosity=verbosity, kwargs...),
        () -> _spectral_moment(prob, T, p, ClassicalSD(2)),   # kwargs NOT forwarded
        prob, T, p, m, verbosity)
_spectral_moment(prob, T, p, m::ClassicalSD; verbosity::Integer=1, kwargs...) =
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

_stationary_var(prob, T, p, m::Collocation; verbosity::Integer=1, kwargs...) =
    _collocation_or_fallback(
        () -> fixPointOfMapping_collocation(prob, T, p; S=m.S,
                                            verbosity=verbosity, kwargs...)[1, 1],
        () -> _stationary_var(prob, T, p, ClassicalSD(2)),    # kwargs NOT forwarded
        prob, T, p, m, verbosity)
function _stationary_var(prob, T, p, m::ClassicalSD; verbosity::Integer=1,
                         tol=nothing, krylovdim=nothing, force=nothing, kwargs...)
    # courtesy: `tol` (the collocation/KrylovKit convention) translates to the
    # gmres `reltol` of the factored fixpoint; `krylovdim`/`force` have no
    # classical-variance counterpart and are ignored rather than crashing gmres
    rst = _classical_result(prob, T, p, m.q; additive=true)
    tol === nothing ? fixPointOfMapping_MF_factored(rst; kwargs...)[1] :
                      fixPointOfMapping_MF_factored(rst; reltol=tol, kwargs...)[1]
end

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
