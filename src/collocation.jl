# High-order Gauss–Legendre collocation solver — user-facing wrappers.
# Adapts an `LDDEProblem` to the internal collocation engines (collocation_engine.jl)
# and exposes second-moment stability / stationary-covariance entry points.
#
# Engine dispatch (automatic, warning-controlled by `verbosity`):
#   constant τ, aligned (τ = r·Δt)  → v9/v8 aligned engine, order 2S
#   constant τ, incommensurate      → vT fractional-limit engine, order [S+1, 2S]
#   function-valued smooth τ(t)     → vT fractional-limit engine, floor S+1
# Delayed multiplicative noise (β ≢ 0) is handled in every case: the aligned
# engine keeps the unpruned block (v8), and the vT engine adds point-sample DOFs
# at the delayed reading positions (vT-full), and multiple delays via per-delay
# history DOFs. Scope: a single Wiener channel; the vT engine additionally
# requires each τ_j(t) ≥ Δt, T-periodic, with a uniformly increasing reading map.

# Build the engine problem from an `LDDEProblem` over one principal period:
# `Prob` for a single constant delay, `ProbT` (g delays) otherwise. The delayed
# multiplicative-noise terms `prob.βs` are paired with the drift delays `prob.Bs`
# by index (β_j shares τ_j with B_j) — the natural regenerative construction;
# `βs` may be shorter (missing ⇒ that delay carries no noise).
function _collocation_prob(prob::LDDEProblem, period::Real)
    d = size(prob.A(0.0), 1)
    # Wiener channels: distinct noise identifiers across α/β/σ. Independent
    # channels sum in the Itô isometry; a single channel is the common case.
    nids = Int[]
    for x in prob.αs; push!(nids, x.nID); end
    for x in prob.βs; push!(nids, x.nID); end
    for x in prob.σs; push!(nids, x.nID); end
    chans = sort(unique(nids))
    K = length(chans)
    g = length(prob.Bs)
    g ≥ 1 || error("the collocation solver needs at least one delay term")
    A = t -> Matrix{Float64}(prob.A(t))
    τraws = [prob.Bs[j].τ.τ for j in 1:g]
    # a β delayed-noise term must read at one of the drift delays τ_j (the engine
    # only carries history DOFs for the registered delays); shared τ object or,
    # for constants, exact numeric match.
    same_delay(a, b) = (a === b) ||
        (a isa Real && b isa Real && abs(a - b) ≤ 1e-12 * max(abs(a), 1.0))
    for x in prob.βs
        any(same_delay(x.cMX.τ.τ, τraws[j]) for j in 1:g) ||
            error("a delayed multiplicative-noise term (β, channel $(x.nID)) has a " *
                  "delay τ that matches none of the drift delays B_j — register that " *
                  "delay with a (possibly zero) DelayMX in the Bs list")
    end
    # single Wiener channel + single constant delay → scalar `Prob` (aligned 2S /
    # incommensurate single-channel engines v8/v9, untouched, bit-identical).
    # All noise terms live on the one channel (or none) ⇒ same-channel terms sum
    # (every β reads at the single delay τ, guaranteed by the guard above).
    if K ≤ 1 && g == 1 && (τraws[1] isa Real)
        α = isempty(prob.αs) ? (t -> zeros(d, d)) :
            (t -> sum(Matrix{Float64}(x(t)) for x in prob.αs))
        σ = isempty(prob.σs) ? (t -> zeros(d, 1)) :
            (t -> sum(reshape(Vector{Float64}(x.V(t)), d, :) for x in prob.σs))
        B = t -> Matrix{Float64}(prob.Bs[1](t))
        β = isempty(prob.βs) ? (t -> zeros(d, d)) :
            (t -> sum(Matrix{Float64}(x(t)) for x in prob.βs))
        return Prob(d, float(period), float(τraws[1]), A, B, α, β, σ)
    end
    # otherwise → ProbT: g delays × K channels. Per-channel α/β/σ; each β is paired
    # to the drift delay with an identical τ (the natural regenerative construction
    # reuses one τ object for B_j and β_j). Independent channels sum in the noise
    # injection. Noise-free ⇒ one trivial zero channel.
    τfs = Function[]; τmins = Float64[]; τmaxs = Float64[]; Bs = Function[]
    for j in 1:g
        τr = τraws[j]
        τf = τr isa Real ? (let v = float(τr); t -> v end) : (t -> float(τr(t)))
        τsamp = τf.(range(0.0, float(period); length=129))
        all(isfinite, τsamp) ||
            error("delay τ_$j(t) returned a non-finite value on [0, T]")
        push!(τfs, τf); push!(τmins, minimum(τsamp)); push!(τmaxs, maximum(τsamp))
        # preserve the user's return type (an SMatrix stays stack-allocated) so the
        # SMatrix noise-operator fast path stays allocation-free; the generic path
        # converts with Matrix(...) as before.
        push!(Bs, t -> prob.Bs[j](t))
    end
    cids = isempty(chans) ? Int[typemin(Int)] : chans   # trivial channel if noise-free
    chanidx = Dict(cid => i for (i, cid) in enumerate(cids))
    # bucket each β term into (channel, first drift delay of matching τ) exactly once
    # — summing multiples and never double-counting when two B_j happen to share a τ
    # (every β is guaranteed to match some delay by the guard above).
    βbucket = [[Any[] for _ in 1:g] for _ in cids]
    for x in prob.βs
        j = findfirst(jj -> same_delay(x.cMX.τ.τ, τraws[jj]), 1:g)
        push!(βbucket[chanidx[x.nID]][j], x)
    end
    αs = Function[]; σs = Function[]; βss = Vector{Function}[]
    for (ci, cid) in enumerate(cids)
        αterms = [x for x in prob.αs if x.nID == cid]
        push!(αs, isempty(αterms) ? (t -> zeros(d, d)) :
                  (t -> sum(Matrix{Float64}(x(t)) for x in αterms)))
        σterms = [x for x in prob.σs if x.nID == cid]
        push!(σs, isempty(σterms) ? (t -> zeros(d, 1)) :
                  (t -> sum(reshape(Vector{Float64}(x.V(t)), d, :) for x in σterms)))
        βrow = Function[]
        for j in 1:g
            βterms = βbucket[ci][j]
            push!(βrow, isempty(βterms) ? (t -> zeros(d, d)) :
                        (t -> sum(Matrix{Float64}(x(t)) for x in βterms)))
        end
        push!(βss, βrow)
    end
    ProbT(d, float(period), τfs, τmins, τmaxs, A, Bs, αs, βss, σs)
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
        @warn "using the fractional-limit collocation engine ($(_ndelays(pb)) " *
              "delay(s), function-valued and/or misaligned) — order floor $(S+1) " *
              "(observed in [$(S+1), $(2S)]); the 2S superconvergence of the aligned " *
              "single-constant-delay engine does not apply. (suppress with verbosity=0)" maxlog=1
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
    ``\\tau``, and ``\\xi(t)=t-\\tau(t)`` uniformly increasing. Delayed reads of
    **rough** (Wiener-driven) components carry no extra order penalty — the
    delayed drift is kept exact in pre-integrated history DOFs.

Delayed **multiplicative** noise (``\\beta \\not\\equiv 0``) is supported in every
case: the aligned engine keeps the unpruned block, and the time-varying engine
adds point-sample DOFs at the delayed reading positions (their covariance filled
from the same causal kernel), so rough delayed *noise* reads keep the order too.
**Multiple delays** ``\\sum_j \\mathbf{B}_j\\mathbf{x}(t-\\tau_j(t))`` (single Wiener
channel) are handled by per-delay integrated-history DOFs sharing the point-sample
states; the ``\\beta_j`` pair with the ``\\mathbf{B}_j`` by index.

Compared with the classical semi-discretization solvers
([`spectralRadiusOfMapping_MF`](@ref) and friends, which are first order in the
second moment), the collocation blocks reach high order at a much smaller memory
footprint per accuracy, at the cost of more covariance sub-blocks per delay slot,
which restricts the method to low/moderate state dimension.

`verbosity=0` suppresses the engine-selection warnings. `force` is accepted for
backward compatibility (no longer meaningful — no noise term is ever dropped).
Extra `kwargs` (`tol`, `krylovdim`) go to the KrylovKit eigensolver.

Supports multiple delays (single Wiener channel).
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
