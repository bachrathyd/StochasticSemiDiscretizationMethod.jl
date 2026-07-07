using SparseArrays
using LinearAlgebra
using StaticArrays
using Arpack
using IterativeSolvers
using KrylovKit

# Multiplication-Free Stochastic Semi-Discretization Method

"""
    MFCoefficients{d, L, L2, Ld}

Holds the precomputed discretized coefficients for the Multiplication-Free SSDM.
- `det`: Deterministic step matrices A_k for each step.
- `stoch_op`: Precomputed stochastic operator matrices E[G_k ⊗ G_l] for the second moment.
- `detV`: Deterministic additive vector f_n for each step.
- `stochV`: Integrated integrated noise variance E[g_n g_n'] for each step.
- `stochGV`: Cross-terms between state coefficients and integrated noise E[G_k y_k g_n'].
"""
struct MFCoefficients{d, L, L2, Ld}
    det::Vector{Vector{Tuple{SMatrix{d, d, Float64, L}, Int}}}
    stoch_op::Vector{Vector{Tuple{SMatrix{L, L, Float64, L2}, Int, Int}}}
    detV::Vector{SVector{d, Float64}}
    stochV::Vector{SMatrix{d, d, Float64, L}}
    stochGV::Vector{Vector{Tuple{SMatrix{L, d, Float64, Ld}, Int, Int}}}
end

"""
    stDiscreteMapping_M2_MF

Mapping object for the second moment Multiplication-Free SSDM.
"""
struct stDiscreteMapping_M2_MF{tT, mx1T, mx12T, v1T, v2T, rstT, coeffT}
    ts::Vector{tT}
    M1_MXs::Vector{mx1T}
    M1_Vs::Vector{v1T}
    M1toM2_MXs::Vector{mx12T}
    M2_Vs::Vector{v2T}
    rst::rstT
    coeffs::coeffT
end

"""
    MFWorkspace{d, L}

Pre-allocated buffers to minimize GC pressure during iterative mapping application.
"""
mutable struct MFWorkspace{d, L}
    C::Matrix{SMatrix{d, d, Float64, L}}
    v::Vector{SVector{d, Float64}}
    C_next_m::Vector{SMatrix{d, d, Float64, L}}
    v_in_zero::Vector{Float64}
end

function MFWorkspace(rst::AbstractResult{d}) where d
    r = div(rst.n, d) - 1
    L = d*d
    C = [zero(SMatrix{d, d, Float64, L}) for i in 1:(r+1), j in 1:(r+1)]
    v = [zero(SVector{d, Float64}) for i in 1:(r+1)]
    C_next_m = Vector{SMatrix{d, d, Float64, L}}(undef, r + 1)
    v_in_zero = zeros((r+1)*d)
    return MFWorkspace{d, L}(C, v, C_next_m, v_in_zero)
end

function get_all_coefficients(rst::AbstractResult{d}) where d
    p = rst.n_steps
    L = d*d
    L2 = L*L
    Ld = L*d
    
    # 1. Deterministic coefficients
    det_all = [Tuple{SMatrix{d, d, Float64, L}, Int}[] for _ in 1:p]
    for (delay_type, submxs) in enumerate(rst.subMXs)
        for i in 1:p
            smx = submxs[i]
            for (range_idx, (r_target, r_source)) in enumerate(smx.ranges)
                k = Int((r_source.start - 1) / d)
                M = smx.MXs[range_idx]
                push!(det_all[i], (SMatrix{d,d,Float64,L}(M), k))
            end
        end
    end
    
    # 2. Stochastic coefficients
    K = length(rst.itoisometrymethod)
    stoch_raw = [Tuple{SMatrix{d, d, SVector{K, Float64}, L}, Int, Int}[] for _ in 1:p]
    for (noise_type, stsubmxs) in enumerate(rst.stsubMXs)
        for i in 1:p
            stsmx = stsubmxs[i]
            w = stsmx.nID
            for (range_idx, (r_target, r_source)) in enumerate(stsmx.ranges)
                k = Int((r_source.start - 1) / d)
                G = stsmx.MXfun[range_idx] 
                push!(stoch_raw[i], (SMatrix{d,d,SVector{K,Float64},L}(G), k, w))
            end
        end
    end

    # Pre-contract Ito Isometry for E[Gk ⊗ Gl]
    stoch_op = [Tuple{SMatrix{L, L, Float64, L2}, Int, Int}[] for _ in 1:p]
    for i in 1:p
        step_stoch = stoch_raw[i]
        for (Gk, k, wk) in step_stoch
            for (Gl, l, wl) in step_stoch
                if wk == wl
                    E_mat_M = zeros(MMatrix{L, L, Float64, L2})
                    for a in 1:d, b in 1:d, c in 1:d, dd in 1:d
                        # vec(Gk C Gl') = (Gl ⊗ Gk) vec(C)
                        row = a + (b-1)*d
                        col = c + (dd-1)*d
                        E_mat_M[row, col] = rst.itoisometrymethod(Gk[a,c], Gl[b,dd])
                    end
                    push!(stoch_op[i], (SMatrix{L,L,Float64,L2}(E_mat_M), k, l))
                end
            end
        end
    end

    # 3. Additive terms
    detV = [SVector{d, Float64}(zeros(d)) for _ in 1:p]
    if rst.calculate_additive && !isempty(rst.subVs)
        for i in 1:p
            detV[i] = SVector{d, Float64}(rst.subVs[i].V)
        end
    end

    stochV = [zeros(SMatrix{d, d, Float64, L}) for _ in 1:p]
    stochGV = [Tuple{SMatrix{L, d, Float64, Ld}, Int, Int}[] for _ in 1:p]
    
    if rst.calculate_additive && !isempty(rst.stsubVs)
        for i in 1:p
            res_noise = zeros(MMatrix{d, d, Float64, L})
            for stsubv_list in rst.stsubVs
                stsv = stsubv_list[i]
                w = stsv.nID
                for a in 1:d, b in 1:d
                    res_noise[a, b] += rst.itoisometrymethod(stsv.Vfun[a], stsv.Vfun[b])
                end
                
                # Cross terms E[G_ac * g_b]
                for stsmx_list in rst.stsubMXs
                    stsmx = stsmx_list[i]
                    if stsmx.nID == w
                        for (range_idx, (r_target, r_source)) in enumerate(stsmx.ranges)
                            k = Int((r_source.start - 1) / d)
                            Gfun = stsmx.MXfun[range_idx]
                            gfun = stsv.Vfun
                            
                            E_Gg_M = zeros(MMatrix{L, d, Float64, Ld})
                            for a in 1:d, b in 1:d, c in 1:d
                                E_Gg_M[a + (b-1)*d, c] = rst.itoisometrymethod(Gfun[a, c], gfun[b])
                            end
                            push!(stochGV[i], (SMatrix{L, d, Float64, Ld}(E_Gg_M), k, w))
                        end
                    end
                end
            end
            stochV[i] = SMatrix{d,d,Float64,L}(res_noise)
        end
    end

    return MFCoefficients{d, L, L2, Ld}(det_all, stoch_op, detV, stochV, stochGV)
end

"""
    DiscreteMapping_M2_MF(rst::AbstractResult)
    DiscreteMapping_M2_MF(prob::LDDEProblem, method, DiscretizationLength; n_steps, calculate_additive=false)

Build the **multiplication-free** representation of the one-period second-moment
map of a stochastic delay problem. Unlike [`DiscreteMapping_M2`](@ref), no
``D\\times D`` operator is assembled; the returned object stores the per-step
coefficients and exposes only the operator *action*, which the matrix-free
solvers [`spectralRadiusOfMapping_MF`](@ref) and [`fixPointOfMapping_MF`](@ref)
consume.

Pass `calculate_additive = true` (via the `LDDEProblem` convenience form or
through `calculateResults`) to enable the stationary-variance fixed point.
"""
function DiscreteMapping_M2_MF(rst::AbstractResult)
    dm1 = DiscreteMapping_M1(rst)
    coeffs = get_all_coefficients(rst)
    M1toM2_MXs = typeof(dm1.M1_MXs)()
    M2_Vs = typeof(dm1.M1_Vs)()
    return stDiscreteMapping_M2_MF(dm1.ts, dm1.M1_MXs, dm1.M1_Vs, M1toM2_MXs, M2_Vs, rst, coeffs)
end

function DiscreteMapping_M2_MF(LDDEP::LDDEProblem, method::DiscretizationMethod, DiscretizationLength::Real; args...)
    DiscreteMapping_M2_MF(StochasticSemiDiscretizationMethod.calculateResults(LDDEP, method, DiscretizationLength; args...))
end

"""
    apply_mapping_M1_MF!(ws, rst, coeffs, v_in; include_additive)

Applies the first-moment mapping using the Multiplication-Free method.
"""
function apply_mapping_M1_MF!(ws::MFWorkspace{d, L}, rst::AbstractResult{d}, coeffs::MFCoefficients{d, L, L2, Ld}, v_in::AbstractVector; include_additive=false) where {d, L, L2, Ld}
    r = div(rst.n, d) - 1
    p = rst.n_steps
    v = ws.v
    c_idx(n) = mod(n, r+1) + 1
    for i in 0:r
        v[c_idx(-i)] = SVector{d, Float64}(v_in[i*d+1:(i+1)*d])
    end
    for n in 0:p-1
        next_n = n + 1
        res = zeros(MVector{d, Float64})
        for (A, k) in coeffs.det[next_n]
            res .+= A * v[c_idx(n-k)]
        end
        if include_additive
            res .+= coeffs.detV[next_n]
        end
        v[c_idx(next_n)] = SVector{d, Float64}(res)
    end
    v_out = zeros(length(v_in))
    for i in 0:r
        v_out[i*d+1:(i+1)*d] = v[c_idx(p-i)]
    end
    return v_out
end

"""
    apply_mapping_M2_MF!(ws, rst, coeffs, m_in, v_in; include_additive)

Applies the second-moment mapping using the Multiplication-Free method.
"""
# Size-gated threading: fine-grained per-step loops only benefit from threads
# when the window is large; below the threshold the task-spawn overhead
# (∝ nthreads, paid 2p times per mapping) exceeds the work.
@inline function _tforeach(f::F, rng; minlen::Int=512) where {F}
    if length(rng) >= minlen && Threads.nthreads() > 1
        Threads.@threads :static for x in rng
            f(x)
        end
    else
        for x in rng
            f(x)
        end
    end
    return nothing
end

function apply_mapping_M2_MF!(ws::MFWorkspace{d, L}, rst::AbstractResult{d}, coeffs::MFCoefficients{d, L, L2, Ld}, m_in::AbstractVector, v_in::AbstractVector; include_additive=false) where {d, L, L2, Ld}
    r = div(rst.n, d) - 1
    p = rst.n_steps
    idx = StochasticSemiDiscretizationMethod.CovVecIdx((r + 1) * d)

    C = ws.C
    v = ws.v
    C_next_m = ws.C_next_m
    c_idx(n) = mod(n, r+1) + 1

    # 0. Initialize (threaded over window blocks; allocation-free SMatrix build)
    _tforeach(0:r) do i
        v[c_idx(-i)] = SVector{d, Float64}(ntuple(q -> v_in[i*d+q], Val(d)))
        ci = c_idx(-i)
        for j in 0:r
            Mat = SMatrix{d,d,Float64,L}(ntuple(q -> begin
                r_row = (q-1) % d + 1; r_col = (q-1) ÷ d + 1
                m_in[idx(i*d + r_row, j*d + r_col)]
            end, Val(L)))
            C[ci, c_idx(-j)] = Mat
        end
    end

    # 1. Propagate
    for n in 0:p-1
        next_n = n + 1
        det_step = coeffs.det[next_n]
        stoch_op = coeffs.stoch_op[next_n]
        detV = coeffs.detV[next_n]

        # v(n+1)
        v_next_S = zero(SVector{d, Float64})
        for (A, k) in det_step
            v_next_S += A * v[c_idx(n-k)]
        end
        if include_additive
            v_next_S += detV
        end

        # C(n+1, m) — stack-allocated accumulation. NOTE: deliberately serial —
        # per-step work is O(r·d³) ≈ tens of µs for d=2, far below the
        # fork-join barrier cost of a threaded region (measured 3× slowdown
        # when threaded on a 36-core box). Only per-APPLY O(r²) loops thread.
        for i in 1:(r+1)
            m = n - r + (i - 1)
            cm = c_idx(m)
            res = zero(SMatrix{d, d, Float64, L})
            for (A, k) in det_step
                res += A * C[c_idx(n-k), cm]
            end
            if include_additive
                res += detV * v[cm]'
            end
            C_next_m[i] = res
        end

        # C(n+1, n+1)
        res_diag = zero(SMatrix{d, d, Float64, L})
        for (Ak, k) in det_step, (Al, l) in det_step
            res_diag += Ak * C[c_idx(n-k), c_idx(n-l)] * Al'
        end

        for (E_mat, k, l) in stoch_op
            Ckl_vec = SVector{L, Float64}(C[c_idx(n-k), c_idx(n-l)])
            res_vec = E_mat * Ckl_vec
            res_diag += SMatrix{d, d, Float64, L}(res_vec)
        end

        if include_additive
            Fv = v_next_S - detV
            res_diag += detV * Fv' + Fv * detV' + detV * detV'
            res_diag += coeffs.stochV[next_n]
            for (E_Gg_mat, k, w) in coeffs.stochGV[next_n]
                term_vec = E_Gg_mat * v[c_idx(n-k)]
                term = SMatrix{d, d, Float64, L}(term_vec)
                res_diag += term + term'
            end
        end

        # Update (serial — same granularity argument as the compute loop)
        v[c_idx(next_n)] = v_next_S
        cn = c_idx(next_n)
        for i in 1:(r+1)
            m = n - r + (i - 1)
            cm = c_idx(m)
            C[cn, cm] = C_next_m[i]
            C[cm, cn] = C_next_m[i]'
        end
        C[cn, cn] = res_diag
    end

    # 2. Extract (threaded over rows)
    m_out = zeros(length(m_in))
    _tforeach(0:r) do i
        p_i_idx = c_idx(p-i)
        for j in 0:r
            Mat = C[p_i_idx, c_idx(p-j)]
            for r_row in 1:d, r_col in 1:d
                vi = i*d + r_row
                vj = j*d + r_col
                if vi <= vj
                    m_out[idx(vi, vj)] = Mat[r_row, r_col]
                end
            end
        end
    end
    return m_out
end

# --- Mapping Operators ---

struct MFMappingOperator{ResultT, CoeffT, WsT} <: AbstractMatrix{Float64}
    rst::ResultT
    coeffs::CoeffT
    D::Int
    ws::WsT
end
Base.size(op::MFMappingOperator) = (op.D, op.D)
Base.size(op::MFMappingOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFMappingOperator) = Float64
LinearAlgebra.issymmetric(op::MFMappingOperator) = false
LinearAlgebra.ishermitian(op::MFMappingOperator) = false

function LinearAlgebra.mul!(y::AbstractVector, op::MFMappingOperator, x::AbstractVector)
    y .= apply_mapping_M2_MF!(op.ws, op.rst, op.coeffs, x, op.ws.v_in_zero, include_additive=false)
end

struct M1MFMappingOperator{ResultT, CoeffT, WsT} <: AbstractMatrix{Float64}
    rst::ResultT
    coeffs::CoeffT
    D::Int
    ws::WsT
end
Base.size(op::M1MFMappingOperator) = (op.D, op.D)
Base.size(op::M1MFMappingOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::M1MFMappingOperator) = Float64

function LinearAlgebra.mul!(y::AbstractVector, op::M1MFMappingOperator, x::AbstractVector)
    y .= apply_mapping_M1_MF!(op.ws, op.rst, op.coeffs, x, include_additive=false)
end

# --- User API ---

"""
    spectralRadiusOfMapping_MF(dm::DiscreteMapping_M2_MF; solver=:KrylovKit, kwargs...) -> Float64

Second-moment spectral radius ``\\rho(\\mathcal{H})`` of the one-period map,
computed **multiplication-free**: the period operator is never assembled, only
its action on a covariance vector is evaluated inside a matrix-free Krylov
iteration. This is the ``\\mathcal{O}(p^2)`` path (in the number of steps ``p``
per period) that replaces the ``\\mathcal{O}(p^4)`` explicit product of
[`DiscreteMapping_M2`](@ref). Mean-square stability corresponds to
``\\rho(\\mathcal{H}) < 1``.

`solver` selects the eigensolver backend (`:KrylovKit` by default, or `:Arpack`);
extra keyword arguments are forwarded to it (e.g. `krylovdim`, `tol`). See
[`spectralRadiusOfMapping_MF_factored`](@ref) for the Kronecker-factored variant
that also removes the ``\\mathcal{O}(d^4)`` state-dimension cost, and
[`spectralRadiusOfMapping_GPU`](@ref) for the CUDA backend.
"""
function spectralRadiusOfMapping_MF(dm::stDiscreteMapping_M2_MF; solver=:KrylovKit, args...)
    # coeffs.det[1][1][1] is the first SMatrix in the first step
    d = size(dm.coeffs.det[1][1][1], 1)
    r = div(dm.rst.n, d) - 1
    D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
    ws = MFWorkspace(dm.rst)
    op = MFMappingOperator(dm.rst, dm.coeffs, D, ws)
    
    if solver == :Arpack
        res, _ = eigs(op, v0=rand(D); args...)
        return abs(res[1])
    elseif solver == :KrylovKit
        vals, _, _ = eigsolve(op, rand(D), 1, :LM; args...)
        return abs(vals[1])
    else
        error("Unknown solver: $solver. Use :Arpack or :KrylovKit.")
    end
end

struct IMinusPhiOperator{OpT} <: AbstractMatrix{Float64}
    op::OpT
end
Base.size(mop::IMinusPhiOperator) = size(mop.op)
Base.size(mop::IMinusPhiOperator, i::Int) = size(mop.op, i)
Base.eltype(mop::IMinusPhiOperator) = Float64
function LinearAlgebra.mul!(y::AbstractVector, mop::IMinusPhiOperator, x::AbstractVector)
    mul!(y, mop.op, x)
    y .= x .- y
end

"""
    fixPointOfMapping_MF(dm::DiscreteMapping_M2_MF; kwargs...) -> Vector{Float64}

Stationary second moment (the fixed point ``\\mathbf{M}^\\ast`` of the one-period
covariance map) computed multiplication-free, for a mean-square stable system
driven by additive noise. Returns the stationary covariance in half-vectorized
(``\\operatorname{vech}``) coordinates; use [`VecToCovMx`](@ref) to reshape it
into a covariance matrix. The leading entry is the stationary variance of the
first state component.

Requires the mapping to have been built with `calculate_additive = true`. See
[`fixPointOfMapping_MF_factored`](@ref) for the Kronecker-factored variant.
"""
function fixPointOfMapping_MF(dm::stDiscreteMapping_M2_MF; args...)
    d = size(dm.coeffs.det[1][1][1], 1)
    r = div(dm.rst.n, d) - 1
    D1 = (r+1)*d
    D2 = StochasticSemiDiscretizationMethod.CovVecIdx(D1).sectionStarts[end]
    ws = MFWorkspace(dm.rst)
    
    op1 = M1MFMappingOperator(dm.rst, dm.coeffs, D1, ws)
    k1 = apply_mapping_M1_MF!(ws, dm.rst, dm.coeffs, zeros(D1), include_additive=true)
    v_star = gmres(IMinusPhiOperator(op1), k1; reltol=1e-15, args...)
    
    k2 = apply_mapping_M2_MF!(ws, dm.rst, dm.coeffs, zeros(D2), v_star, include_additive=true)
    op2 = MFMappingOperator(dm.rst, dm.coeffs, D2, ws)
    m_star = gmres(IMinusPhiOperator(op2), k2; reltol=1e-15, args...)
    return m_star
end
