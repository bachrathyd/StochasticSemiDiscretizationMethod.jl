# =============================================================================
# CUDA backend for the multiplication-free second-moment solver.
# This is a package extension: it is loaded automatically when the user runs
# `using CUDA` alongside StochasticSemiDiscretizationMethod, and it adds the
# GPU methods of `spectralRadiusOfMapping_GPU`, `fixPointOfMapping_GPU`, and
# `spectralRadiusOfMapping_auto`. Without CUDA loaded, the core package keeps
# CPU-only fallbacks (see src/functions_gpu_stubs.jl).
# =============================================================================
module StochasticSemiDiscretizationMethodCUDAExt

using CUDA
using LinearAlgebra
using KrylovKit
import StochasticSemiDiscretizationMethod
import StochasticSemiDiscretizationMethod: stDiscreteMapping_M2_MF, AbstractResult,
    spectralRadiusOfMapping_MF, spectralRadiusOfMapping_GPU, fixPointOfMapping_GPU,
    spectralRadiusOfMapping_auto

# Bind cooperative-group functions as module-level constants so the GPU
# compiler can resolve them without dynamic getproperty dispatch.
const _cg_this_grid = CUDA.CG.this_grid
const _cg_sync      = CUDA.CG.sync

# --- 1. GPU Memory Layout for Coefficients ---

struct MFGPUCoefficients
    det_A::CuArray{Float64, 4}
    det_k::CuArray{Int32, 2}
    num_det::CuArray{Int32, 1}

    stoch_E::CuArray{Float64, 4}
    stoch_k::CuArray{Int32, 2}
    stoch_l::CuArray{Int32, 2}
    num_stoch::CuArray{Int32, 1}

    detV::CuArray{Float64, 2}
    stochV::CuArray{Float64, 3}

    stochGV::CuArray{Float64, 4}
    stochGV_k::CuArray{Int32, 2}
    num_stochGV::CuArray{Int32, 1}
end

function extract_gpu_coeffs(coeffs, p::Int, d::Int, include_additive::Bool)
    L = d*d
    max_det      = max(1, maximum(length.(coeffs.det)))
    max_stoch    = max(1, maximum(length.(coeffs.stoch_op)))
    max_stoch_GV = (include_additive && !isempty(coeffs.stochGV)) ?
                       max(1, maximum(length.(coeffs.stochGV))) : 1

    det_A_cpu    = zeros(Float64, d, d, max_det, p)
    det_k_cpu    = zeros(Int32, max_det, p)
    num_det_cpu  = zeros(Int32, p)

    stoch_E_cpu    = zeros(Float64, L, L, max_stoch, p)
    stoch_k_cpu    = zeros(Int32, max_stoch, p)
    stoch_l_cpu    = zeros(Int32, max_stoch, p)
    num_stoch_cpu  = zeros(Int32, p)

    detV_cpu   = zeros(Float64, d, p)
    stochV_cpu = zeros(Float64, d, d, p)

    stochGV_cpu   = zeros(Float64, L, d, max_stoch_GV, p)
    stochGV_k_cpu = zeros(Int32, max_stoch_GV, p)
    num_stochGV_cpu = zeros(Int32, p)

    for n in 1:p
        step_det = coeffs.det[n]
        num_det_cpu[n] = length(step_det)
        for i in 1:length(step_det)
            Ak, k = step_det[i]
            det_A_cpu[:, :, i, n] .= Ak
            det_k_cpu[i, n] = k
        end

        step_stoch = coeffs.stoch_op[n]
        num_stoch_cpu[n] = length(step_stoch)
        for i in 1:length(step_stoch)
            Ek, k, l = step_stoch[i]
            stoch_E_cpu[:, :, i, n] .= Ek
            stoch_k_cpu[i, n] = k
            stoch_l_cpu[i, n] = l
        end

        if include_additive
            detV_cpu[:, n] .= coeffs.detV[n]
            stochV_cpu[:, :, n] .= coeffs.stochV[n]

            if !isempty(coeffs.stochGV)
                step_stochGV = coeffs.stochGV[n]
                num_stochGV_cpu[n] = length(step_stochGV)
                for i in 1:length(step_stochGV)
                    Ek, k, w = step_stochGV[i]
                    stochGV_cpu[:, :, i, n] .= Ek
                    stochGV_k_cpu[i, n] = k
                end
            end
        end
    end

    return MFGPUCoefficients(
        CuArray(det_A_cpu), CuArray(det_k_cpu), CuArray(num_det_cpu),
        CuArray(stoch_E_cpu), CuArray(stoch_k_cpu), CuArray(stoch_l_cpu), CuArray(num_stoch_cpu),
        CuArray(detV_cpu), CuArray(stochV_cpu),
        CuArray(stochGV_cpu), CuArray(stochGV_k_cpu), CuArray(num_stochGV_cpu)
    )
end

# --- 2. GPU Workspace ---

struct MFGPUWorkspace
    C::CuArray{Float64, 4}
    v::CuArray{Float64, 2}
    C_next_m::CuArray{Float64, 1}
    v_next::CuArray{Float64, 1}
    idx_sectionStarts::CuArray{Int32, 1}
    v_in_zero::CuArray{Float64, 1}
end

function MFGPUWorkspace(rst::AbstractResult{d}) where d
    r = div(rst.n, d) - 1

    C        = CuArray(zeros(Float64, d, d, r+1, r+1))
    v        = CuArray(zeros(Float64, d, r+1))
    C_next_m = CuArray(zeros(Float64, d * d * (r+2)))
    v_next   = CuArray(zeros(Float64, d))
    v_in_zero = CuArray(zeros(Float64, (r+1)*d))

    idx_cpu = Int32.(StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts)
    idx_sectionStarts = CuArray(idx_cpu)

    return MFGPUWorkspace(C, v, C_next_m, v_next, idx_sectionStarts, v_in_zero)
end

# ============================================================
# Helper: circular buffer index  (v1 — while-loop version)
# ============================================================

@inline function gpu_c_idx(n::Int32, rp1::Int32)
    res = n
    while res < 0;    res += rp1; end
    while res >= rp1; res -= rp1; end
    return res + Int32(1)
end

# ============================================================
# V1 Kernels  (original implementation)
# ============================================================

function kernel_M1_MF!(
    v_out, v_in, v, v_next,
    det_A, det_k, num_det, detV,
    p::Int32, r::Int32, d::Int32, include_additive::Bool
)
    @inbounds begin
        tid  = threadIdx().x
        rp1  = r + Int32(1)

        if tid == 1
            for idx_i in Int32(0):r
                phys_idx = gpu_c_idx(-idx_i, rp1)
                for row in Int32(1):d
                    idx_v = idx_i * d + (row - Int32(1))
                    v[row, phys_idx] = v_in[idx_v + 1]
                end
            end
        end
        sync_threads()

        for n in Int32(0):p-Int32(1)
            next_n = n + Int32(1)
            nd = num_det[next_n]
            if tid <= d
                row = tid
                val = 0.0
                for i in Int32(1):nd
                    k = det_k[i, next_n]
                    idx_nk_phys = gpu_c_idx(n - k, rp1)
                    for s in Int32(1):d
                        val += det_A[row, s, i, next_n] * v[s, idx_nk_phys]
                    end
                end
                if include_additive
                    val += detV[row, next_n]
                end
                v_next[row] = val
            end
            sync_threads()
            next_n_phys = gpu_c_idx(next_n, rp1)
            if tid <= d
                v[tid, next_n_phys] = v_next[tid]
            end
            sync_threads()
        end

        if tid == 1
            for idx_i in Int32(0):r
                p_i = p - idx_i
                p_i_phys = gpu_c_idx(p_i, rp1)
                for row in Int32(1):d
                    idx_v = idx_i * d + (row - Int32(1))
                    v_out[idx_v + 1] = v[row, p_i_phys]
                end
            end
        end
    end
    return nothing
end

function kernel_M2_MF!(
    m_out, v_out, m_in, v_in,
    C, v, C_next_m, v_next,
    det_A, det_k, num_det, detV,
    stoch_E, stoch_k, stoch_l, num_stoch, stochV, stoch_GV, stoch_GV_k, num_stoch_GV,
    p::Int32, r::Int32, d::Int32, include_additive::Bool,
    idx_sectionStarts
)
    @inbounds begin
        tid    = threadIdx().x
        stride = blockDim().x
        rp1    = r + Int32(1)

        # --- 0. Initialize (serial: tid==1) ---
        if tid == 1
            for idx_i in Int32(0):r
                phys_idx = gpu_c_idx(-idx_i, rp1)
                for row in Int32(1):d
                    idx_v = idx_i * d + (row - Int32(1))
                    v[row, phys_idx] = v_in[idx_v + 1]
                end
            end

            D1 = rp1 * d
            for idx_i in Int32(0):r
                for idx_j in Int32(0):r
                    phys_i = gpu_c_idx(-idx_i, rp1)
                    phys_j = gpu_c_idx(-idx_j, rp1)
                    for row in Int32(1):d
                        for col in Int32(1):d
                            vi = idx_i * d + row
                            vj = idx_j * d + col
                            if vi <= D1 && vj <= D1
                                min_v = vi < vj ? vi : vj
                                max_v = vi > vj ? vi : vj
                                diff  = max_v - min_v
                                idx_1d = idx_sectionStarts[diff + 1] + min_v
                                C[row, col, phys_i, phys_j] = m_in[idx_1d]
                            end
                        end
                    end
                end
            end
        end
        sync_threads()

        # --- 1. Propagate ---
        for n in Int32(0):p-Int32(1)
            next_n = n + Int32(1)
            nd = num_det[next_n]
            ns = num_stoch[next_n]

            if tid <= d
                row = tid
                val = 0.0
                for i in Int32(1):nd
                    k = det_k[i, next_n]
                    idx_nk_phys = gpu_c_idx(n - k, rp1)
                    for s in Int32(1):d
                        val += det_A[row, s, i, next_n] * v[s, idx_nk_phys]
                    end
                end
                if include_additive; val += detV[row, next_n]; end
                v_next[row] = val
            end

            curr_idx = 1
            for m_offset in Int32(0):r
                for col in Int32(1):d
                    for row in Int32(1):d
                        if curr_idx % stride == tid % stride
                            m = (n - r) + m_offset
                            idx_m_phys = gpu_c_idx(m, rp1)
                            val = 0.0
                            for i in Int32(1):nd
                                k = det_k[i, next_n]
                                idx_nk_phys = gpu_c_idx(n - k, rp1)
                                for s in Int32(1):d
                                    val += det_A[row, s, i, next_n] * C[s, col, idx_nk_phys, idx_m_phys]
                                end
                            end
                            if include_additive
                                val += detV[row, next_n] * v[col, idx_m_phys]
                            end
                            flat_idx = m_offset * d * d + (col - Int32(1)) * d + row
                            C_next_m[flat_idx] = val
                        end
                        curr_idx += 1
                    end
                end
            end

            curr_idx = 1
            for col in Int32(1):d
                for row in Int32(1):d
                    if curr_idx % stride == tid % stride
                        val = 0.0
                        for i in Int32(1):nd
                            k = det_k[i, next_n]
                            idx_nk_phys = gpu_c_idx(n - k, rp1)
                            for j in Int32(1):nd
                                l = det_k[j, next_n]
                                idx_nl_phys = gpu_c_idx(n - l, rp1)
                                for s in Int32(1):d
                                    for t in Int32(1):d
                                        val += det_A[row, s, i, next_n] * C[s, t, idx_nk_phys, idx_nl_phys] * det_A[col, t, j, next_n]
                                    end
                                end
                            end
                        end
                        lin_idx = row + (col - Int32(1)) * d
                        for i in Int32(1):ns
                            k = stoch_k[i, next_n]
                            l = stoch_l[i, next_n]
                            idx_nk_phys = gpu_c_idx(n - k, rp1)
                            idx_nl_phys = gpu_c_idx(n - l, rp1)
                            for s in Int32(1):d
                                for t in Int32(1):d
                                    lin_idx_inner = s + (t - Int32(1)) * d
                                    val += stoch_E[lin_idx, lin_idx_inner, i, next_n] * C[s, t, idx_nk_phys, idx_nl_phys]
                                end
                            end
                        end
                        if include_additive
                            Fv_row = v_next[row] - detV[row, next_n]
                            Fv_col = v_next[col] - detV[col, next_n]
                            val += detV[row, next_n] * Fv_col + Fv_row * detV[col, next_n] + detV[row, next_n] * detV[col, next_n]
                            val += stochV[row, col, next_n]
                            ns_gv = num_stoch_GV[next_n]
                            for i in Int32(1):ns_gv
                                k = stoch_GV_k[i, next_n]
                                idx_nk_phys = gpu_c_idx(n - k, rp1)
                                term_row_col = 0.0
                                term_col_row = 0.0
                                for s in Int32(1):d
                                    term_row_col += stoch_GV[lin_idx, s, i, next_n] * v[s, idx_nk_phys]
                                    lin_idx_T     = col + (row - Int32(1)) * d
                                    term_col_row += stoch_GV[lin_idx_T, s, i, next_n] * v[s, idx_nk_phys]
                                end
                                val += term_row_col + term_col_row
                            end
                        end
                        flat_idx = (rp1) * d * d + (col - Int32(1)) * d + row
                        C_next_m[flat_idx] = val
                    end
                    curr_idx += 1
                end
            end

            sync_threads()

            next_n_phys = gpu_c_idx(next_n, rp1)
            if tid <= d
                v[tid, next_n_phys] = v_next[tid]
            end

            curr_idx = 1
            for m_offset in Int32(0):r
                for col in Int32(1):d
                    for row in Int32(1):d
                        if curr_idx % stride == tid % stride
                            m = (n - r) + m_offset
                            idx_m_phys = gpu_c_idx(m, rp1)
                            flat_idx = m_offset * d * d + (col - Int32(1)) * d + row
                            val = C_next_m[flat_idx]
                            C[row, col, next_n_phys, idx_m_phys] = val
                            C[col, row, idx_m_phys, next_n_phys] = val
                        end
                        curr_idx += 1
                    end
                end
            end

            curr_idx = 1
            for col in Int32(1):d
                for row in Int32(1):d
                    if curr_idx % stride == tid % stride
                        flat_idx = (rp1) * d * d + (col - Int32(1)) * d + row
                        C[row, col, next_n_phys, next_n_phys] = C_next_m[flat_idx]
                    end
                    curr_idx += 1
                end
            end

            sync_threads()
        end

        # --- 2. Extract (serial: tid==1) ---
        if tid == 1
            for idx_i in Int32(0):r
                for idx_j in Int32(0):r
                    p_i = p - idx_i
                    p_i_phys = gpu_c_idx(p_i, rp1)
                    p_j = p - idx_j
                    p_j_phys = gpu_c_idx(p_j, rp1)
                    for row in Int32(1):d
                        for col in Int32(1):d
                            vi = idx_i * d + row
                            vj = idx_j * d + col
                            if vi <= vj
                                val   = C[row, col, p_i_phys, p_j_phys]
                                diff  = vj - vi
                                idx_1d = idx_sectionStarts[diff + 1] + vi
                                m_out[idx_1d] = val
                            end
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# O(1) circular-buffer index (replaces the while-loop version in hot paths)
@inline function gpu_c_idx_v2(n::Int32, rp1::Int32)
    r = n % rp1
    return (r < Int32(0) ? r + rp1 : r) + Int32(1)
end

# --- 4. Mapping Operators ---

struct M1MFGPUMappingOperator{WsT} <: AbstractMatrix{Float64}
    coeffs::MFGPUCoefficients
    D::Int; r::Int; d::Int; p::Int
    ws::WsT
end
Base.size(op::M1MFGPUMappingOperator) = (op.D, op.D)
Base.size(op::M1MFGPUMappingOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::M1MFGPUMappingOperator) = Float64
LinearAlgebra.issymmetric(op::M1MFGPUMappingOperator) = false
LinearAlgebra.ishermitian(op::M1MFGPUMappingOperator) = false

function LinearAlgebra.mul!(y::AbstractVector, op::M1MFGPUMappingOperator, x::AbstractVector)
    @cuda threads=1024 blocks=1 kernel_M1_MF!(
        y, x, op.ws.v, op.ws.v_next,
        op.coeffs.det_A, op.coeffs.det_k, op.coeffs.num_det, op.coeffs.detV,
        Int32(op.p), Int32(op.r), Int32(op.d), false
    )
    return y
end

struct MFGPUIMinusPhiOperator{OpT} <: AbstractMatrix{Float64}
    op::OpT
end
Base.size(mop::MFGPUIMinusPhiOperator) = size(mop.op)
Base.size(mop::MFGPUIMinusPhiOperator, i::Int) = size(mop.op, i)
Base.eltype(mop::MFGPUIMinusPhiOperator) = Float64
LinearAlgebra.issymmetric(mop::MFGPUIMinusPhiOperator) = false
LinearAlgebra.ishermitian(mop::MFGPUIMinusPhiOperator) = false
function LinearAlgebra.mul!(y::AbstractVector, mop::MFGPUIMinusPhiOperator, x::AbstractVector)
    mul!(y, mop.op, x)
    y .= x .- y
    return y
end

# ============================================================
# V3 Kernel  —  Modification C: multi-block cooperative
#   Same flat-indexed logic as v2, but tid/stride are GLOBAL
#   (across all blocks) and sync is a grid-level barrier.
#   Launched with @cuda cooperative=true threads=256 blocks=N_SM
# ============================================================

function kernel_M2_MF_v3!(
    m_out, m_in,
    C, C_next_m,
    det_A, det_k, num_det,
    stoch_E, stoch_k, stoch_l, num_stoch,
    p::Int32, r::Int32, d::Int32,
    idx_sectionStarts
)
    @inbounds begin
        # Global thread index and total-thread stride
        tid    = Int32(threadIdx().x) + Int32((blockIdx().x - Int32(1)) * blockDim().x)
        stride = Int32(blockDim().x) * Int32(gridDim().x)
        rp1    = r + Int32(1)
        L      = d * d
        total_C     = rp1 * rp1 * L
        total_cross = rp1 * L

        grid = _cg_this_grid()

        # --- 0. Parallel init: unpack m_in → C ---
        for flat in (tid - Int32(1)):stride:(total_C - Int32(1))
            idx_j  = flat ÷ (rp1 * L)
            rem1   = flat - idx_j * (rp1 * L)
            idx_i  = rem1 ÷ L
            rem2   = rem1 - idx_i * L
            col    = rem2 ÷ d + Int32(1)
            row    = rem2 - (col - Int32(1)) * d + Int32(1)
            phys_i = gpu_c_idx_v2(-idx_i, rp1)
            phys_j = gpu_c_idx_v2(-idx_j, rp1)
            vi = idx_i * d + row
            vj = idx_j * d + col
            min_v  = vi < vj ? vi : vj
            max_v  = vi > vj ? vi : vj
            diff   = max_v - min_v
            C[row, col, phys_i, phys_j] = m_in[idx_sectionStarts[diff + Int32(1)] + min_v]
        end

        _cg_sync(grid)

        # --- 1. Propagate ---
        for n in Int32(0):p - Int32(1)
            next_n = n + Int32(1)
            nd = num_det[next_n]
            ns = num_stoch[next_n]

            for flat in (tid - Int32(1)):stride:(total_cross - Int32(1))
                m_offset   = flat ÷ L
                rem        = flat - m_offset * L
                col        = rem ÷ d + Int32(1)
                row        = rem - (col - Int32(1)) * d + Int32(1)
                idx_m_phys = gpu_c_idx_v2((n - r) + m_offset, rp1)
                val = 0.0
                for i in Int32(1):nd
                    idx_nk = gpu_c_idx_v2(n - det_k[i, next_n], rp1)
                    for s in Int32(1):d
                        val += det_A[row, s, i, next_n] * C[s, col, idx_nk, idx_m_phys]
                    end
                end
                C_next_m[flat + Int32(1)] = val
            end

            for flat in (tid - Int32(1)):stride:(L - Int32(1))
                col = flat ÷ d + Int32(1)
                row = flat - (col - Int32(1)) * d + Int32(1)
                lin = row + (col - Int32(1)) * d
                val = 0.0
                for i in Int32(1):nd
                    idx_nk = gpu_c_idx_v2(n - det_k[i, next_n], rp1)
                    for j in Int32(1):nd
                        idx_nl = gpu_c_idx_v2(n - det_k[j, next_n], rp1)
                        for s in Int32(1):d
                            for t in Int32(1):d
                                val += det_A[row, s, i, next_n] * C[s, t, idx_nk, idx_nl] * det_A[col, t, j, next_n]
                            end
                        end
                    end
                end
                for i in Int32(1):ns
                    idx_nk = gpu_c_idx_v2(n - stoch_k[i, next_n], rp1)
                    idx_nl = gpu_c_idx_v2(n - stoch_l[i, next_n], rp1)
                    for s in Int32(1):d
                        for t in Int32(1):d
                            val += stoch_E[lin, s + (t - Int32(1)) * d, i, next_n] * C[s, t, idx_nk, idx_nl]
                        end
                    end
                end
                C_next_m[total_cross + flat + Int32(1)] = val
            end

            _cg_sync(grid)

            next_phys = gpu_c_idx_v2(next_n, rp1)
            for flat in (tid - Int32(1)):stride:(total_cross - Int32(1))
                m_offset   = flat ÷ L
                rem        = flat - m_offset * L
                col        = rem ÷ d + Int32(1)
                row        = rem - (col - Int32(1)) * d + Int32(1)
                idx_m_phys = gpu_c_idx_v2((n - r) + m_offset, rp1)
                val        = C_next_m[flat + Int32(1)]
                C[row, col, next_phys, idx_m_phys] = val
                C[col, row, idx_m_phys, next_phys] = val
            end
            for flat in (tid - Int32(1)):stride:(L - Int32(1))
                col = flat ÷ d + Int32(1)
                row = flat - (col - Int32(1)) * d + Int32(1)
                C[row, col, next_phys, next_phys] = C_next_m[total_cross + flat + Int32(1)]
            end

            _cg_sync(grid)
        end

        # --- 2. Parallel extract: C → m_out ---
        for flat in (tid - Int32(1)):stride:(total_C - Int32(1))
            idx_j  = flat ÷ (rp1 * L)
            rem1   = flat - idx_j * (rp1 * L)
            idx_i  = rem1 ÷ L
            rem2   = rem1 - idx_i * L
            col    = rem2 ÷ d + Int32(1)
            row    = rem2 - (col - Int32(1)) * d + Int32(1)
            vi = idx_i * d + row
            vj = idx_j * d + col
            if vi <= vj
                val  = C[row, col, gpu_c_idx_v2(p - idx_i, rp1), gpu_c_idx_v2(p - idx_j, rp1)]
                diff = vj - vi
                m_out[idx_sectionStarts[diff + Int32(1)] + vi] = val
            end
        end
    end
    return nothing
end

struct MFGPUMappingOperator_v3{WsT} <: AbstractMatrix{Float64}
    coeffs::MFGPUCoefficients
    D::Int; r::Int; d::Int; p::Int; n_sm::Int
    ws::WsT
end
Base.size(op::MFGPUMappingOperator_v3) = (op.D, op.D)
Base.size(op::MFGPUMappingOperator_v3, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFGPUMappingOperator_v3) = Float64
LinearAlgebra.issymmetric(op::MFGPUMappingOperator_v3) = false
LinearAlgebra.ishermitian(op::MFGPUMappingOperator_v3) = false

function LinearAlgebra.mul!(y::AbstractVector, op::MFGPUMappingOperator_v3, x::AbstractVector)
    @cuda threads=256 blocks=op.n_sm cooperative=true kernel_M2_MF_v3!(
        y, x,
        op.ws.C, op.ws.C_next_m,
        op.coeffs.det_A, op.coeffs.det_k, op.coeffs.num_det,
        op.coeffs.stoch_E, op.coeffs.stoch_k, op.coeffs.stoch_l, op.coeffs.num_stoch,
        Int32(op.p), Int32(op.r), Int32(op.d), op.ws.idx_sectionStarts
    )
    return y
end

# ── GPU v4: non-cooperative multi-launch kernels ──────────────────────────────
# Each mul! launches 2p+2 regular (non-cooperative) kernels in the default
# CUDA stream.  Stream ordering provides implicit barriers between steps,
# eliminating the cooperative-launch overhead (~4.6 ms per mul! for p=10).
# Faster than v3 for p ≲ 500; v3 wins for larger p.

function kernel_M2_MF_v4_init!(
    m_in, C,
    r::Int32, d::Int32,
    idx_sectionStarts
)
    @inbounds begin
        tid     = Int32(threadIdx().x) + Int32((blockIdx().x - Int32(1)) * blockDim().x)
        stride  = Int32(blockDim().x) * Int32(gridDim().x)
        rp1     = r + Int32(1)
        L       = d * d
        total_C = rp1 * rp1 * L
        for flat in (tid - Int32(1)):stride:(total_C - Int32(1))
            idx_j  = flat ÷ (rp1 * L)
            rem1   = flat - idx_j * (rp1 * L)
            idx_i  = rem1 ÷ L
            rem2   = rem1 - idx_i * L
            col    = rem2 ÷ d + Int32(1)
            row    = rem2 - (col - Int32(1)) * d + Int32(1)
            phys_i = gpu_c_idx_v2(-idx_i, rp1)
            phys_j = gpu_c_idx_v2(-idx_j, rp1)
            vi     = idx_i * d + row
            vj     = idx_j * d + col
            min_v  = vi < vj ? vi : vj
            max_v  = vi > vj ? vi : vj
            diff   = max_v - min_v
            C[row, col, phys_i, phys_j] = m_in[idx_sectionStarts[diff + Int32(1)] + min_v]
        end
    end
    return nothing
end

function kernel_M2_MF_v4_compute!(
    C_next_m, C,
    det_A, det_k, num_det,
    stoch_E, stoch_k, stoch_l, num_stoch,
    r::Int32, d::Int32, n::Int32
)
    @inbounds begin
        tid         = Int32(threadIdx().x) + Int32((blockIdx().x - Int32(1)) * blockDim().x)
        stride      = Int32(blockDim().x) * Int32(gridDim().x)
        rp1         = r + Int32(1)
        L           = d * d
        total_cross = rp1 * L
        next_n      = n + Int32(1)
        nd          = num_det[next_n]
        ns          = num_stoch[next_n]

        # Cross-terms → C_next_m[1..total_cross]
        for flat in (tid - Int32(1)):stride:(total_cross - Int32(1))
            m_offset   = flat ÷ L
            rem        = flat - m_offset * L
            col        = rem ÷ d + Int32(1)
            row        = rem - (col - Int32(1)) * d + Int32(1)
            idx_m_phys = gpu_c_idx_v2((n - r) + m_offset, rp1)
            val = 0.0
            for i in Int32(1):nd
                idx_nk = gpu_c_idx_v2(n - det_k[i, next_n], rp1)
                for s in Int32(1):d
                    val += det_A[row, s, i, next_n] * C[s, col, idx_nk, idx_m_phys]
                end
            end
            C_next_m[flat + Int32(1)] = val
        end

        # Diagonal (det + stoch) → C_next_m[total_cross+1..total_cross+L]
        for flat in (tid - Int32(1)):stride:(L - Int32(1))
            col = flat ÷ d + Int32(1)
            row = flat - (col - Int32(1)) * d + Int32(1)
            lin = row + (col - Int32(1)) * d
            val = 0.0
            for i in Int32(1):nd
                idx_nk = gpu_c_idx_v2(n - det_k[i, next_n], rp1)
                for j in Int32(1):nd
                    idx_nl = gpu_c_idx_v2(n - det_k[j, next_n], rp1)
                    for s in Int32(1):d
                        for t in Int32(1):d
                            val += det_A[row, s, i, next_n] * C[s, t, idx_nk, idx_nl] * det_A[col, t, j, next_n]
                        end
                    end
                end
            end
            for i in Int32(1):ns
                idx_nk = gpu_c_idx_v2(n - stoch_k[i, next_n], rp1)
                idx_nl = gpu_c_idx_v2(n - stoch_l[i, next_n], rp1)
                for s in Int32(1):d
                    for t in Int32(1):d
                        val += stoch_E[lin, s + (t - Int32(1)) * d, i, next_n] * C[s, t, idx_nk, idx_nl]
                    end
                end
            end
            C_next_m[total_cross + flat + Int32(1)] = val
        end
    end
    return nothing
end

function kernel_M2_MF_v4_writeback!(
    C, C_next_m,
    r::Int32, d::Int32, n::Int32
)
    @inbounds begin
        tid         = Int32(threadIdx().x) + Int32((blockIdx().x - Int32(1)) * blockDim().x)
        stride      = Int32(blockDim().x) * Int32(gridDim().x)
        rp1         = r + Int32(1)
        L           = d * d
        total_cross = rp1 * L
        next_n      = n + Int32(1)
        next_phys   = gpu_c_idx_v2(next_n, rp1)

        # Cross-term writeback — skip m_offset=0 (idx_m_phys == next_phys),
        # which is overwritten by the diagonal loop below; skipping avoids a
        # race between threads for the C[next_phys, next_phys] cells.
        for flat in (tid - Int32(1)):stride:(total_cross - Int32(1))
            m_offset = flat ÷ L
            if m_offset > Int32(0)
                rem        = flat - m_offset * L
                col        = rem ÷ d + Int32(1)
                row        = rem - (col - Int32(1)) * d + Int32(1)
                idx_m_phys = gpu_c_idx_v2((n - r) + m_offset, rp1)
                val        = C_next_m[flat + Int32(1)]
                C[row, col, next_phys, idx_m_phys] = val
                C[col, row, idx_m_phys, next_phys] = val
            end
        end

        # Diagonal writeback — authoritative write for C[next_phys, next_phys]
        for flat in (tid - Int32(1)):stride:(L - Int32(1))
            col = flat ÷ d + Int32(1)
            row = flat - (col - Int32(1)) * d + Int32(1)
            C[row, col, next_phys, next_phys] = C_next_m[total_cross + flat + Int32(1)]
        end
    end
    return nothing
end

function kernel_M2_MF_v4_extract!(
    m_out, C,
    p::Int32, r::Int32, d::Int32,
    idx_sectionStarts
)
    @inbounds begin
        tid     = Int32(threadIdx().x) + Int32((blockIdx().x - Int32(1)) * blockDim().x)
        stride  = Int32(blockDim().x) * Int32(gridDim().x)
        rp1     = r + Int32(1)
        L       = d * d
        total_C = rp1 * rp1 * L
        for flat in (tid - Int32(1)):stride:(total_C - Int32(1))
            idx_j  = flat ÷ (rp1 * L)
            rem1   = flat - idx_j * (rp1 * L)
            idx_i  = rem1 ÷ L
            rem2   = rem1 - idx_i * L
            col    = rem2 ÷ d + Int32(1)
            row    = rem2 - (col - Int32(1)) * d + Int32(1)
            vi     = idx_i * d + row
            vj     = idx_j * d + col
            if vi <= vj
                val  = C[row, col, gpu_c_idx_v2(p - idx_i, rp1), gpu_c_idx_v2(p - idx_j, rp1)]
                diff = vj - vi
                m_out[idx_sectionStarts[diff + Int32(1)] + vi] = val
            end
        end
    end
    return nothing
end

struct MFGPUMappingOperator_v4{WsT} <: AbstractMatrix{Float64}
    coeffs::MFGPUCoefficients
    D::Int; r::Int; d::Int; p::Int; n_blocks::Int
    ws::WsT
end
Base.size(op::MFGPUMappingOperator_v4) = (op.D, op.D)
Base.size(op::MFGPUMappingOperator_v4, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFGPUMappingOperator_v4) = Float64
LinearAlgebra.issymmetric(op::MFGPUMappingOperator_v4) = false
LinearAlgebra.ishermitian(op::MFGPUMappingOperator_v4) = false

function LinearAlgebra.mul!(y::AbstractVector, op::MFGPUMappingOperator_v4, x::AbstractVector)
    nb = op.n_blocks
    @cuda threads=256 blocks=nb kernel_M2_MF_v4_init!(
        x, op.ws.C, Int32(op.r), Int32(op.d), op.ws.idx_sectionStarts)
    for n in Int32(0):Int32(op.p - 1)
        @cuda threads=256 blocks=nb kernel_M2_MF_v4_compute!(
            op.ws.C_next_m, op.ws.C,
            op.coeffs.det_A, op.coeffs.det_k, op.coeffs.num_det,
            op.coeffs.stoch_E, op.coeffs.stoch_k, op.coeffs.stoch_l, op.coeffs.num_stoch,
            Int32(op.r), Int32(op.d), n)
        @cuda threads=256 blocks=nb kernel_M2_MF_v4_writeback!(
            op.ws.C, op.ws.C_next_m, Int32(op.r), Int32(op.d), n)
    end
    @cuda threads=256 blocks=nb kernel_M2_MF_v4_extract!(
        y, op.ws.C, Int32(op.p), Int32(op.r), Int32(op.d), op.ws.idx_sectionStarts)
    return y
end

# ── GPU v4g: CUDA-graph wrapper around v4 ────────────────────────────────────
# Captures the 2p+2 kernel sequence once into a CuGraphExec and replays it
# with a single low-overhead launch, eliminating ~0.27 ms per @cuda call.
# x_buf / y_buf have fixed GPU addresses baked into the graph; mul! copies
# x → x_buf before replay and y_buf → y after.

struct MFGPUMappingOperator_v4g{WsT} <: AbstractMatrix{Float64}
    coeffs::MFGPUCoefficients
    D::Int; r::Int; d::Int; p::Int; n_blocks::Int
    ws::WsT
    x_buf::CuVector{Float64}
    y_buf::CuVector{Float64}
    exec::CuGraphExec
end
Base.size(op::MFGPUMappingOperator_v4g) = (op.D, op.D)
Base.size(op::MFGPUMappingOperator_v4g, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFGPUMappingOperator_v4g) = Float64
LinearAlgebra.issymmetric(op::MFGPUMappingOperator_v4g) = false
LinearAlgebra.ishermitian(op::MFGPUMappingOperator_v4g) = false

function LinearAlgebra.mul!(y::AbstractVector, op::MFGPUMappingOperator_v4g, x::AbstractVector)
    copyto!(op.x_buf, x)
    CUDA.launch(op.exec)
    copyto!(y, op.y_buf)
    return y
end

function _build_v4_graph(op::MFGPUMappingOperator_v4, x_buf::CuVector{Float64},
                          y_buf::CuVector{Float64})
    nb = op.n_blocks
    g = CUDA.capture() do
        @cuda threads=256 blocks=nb kernel_M2_MF_v4_init!(
            x_buf, op.ws.C, Int32(op.r), Int32(op.d), op.ws.idx_sectionStarts)
        for n in Int32(0):Int32(op.p - 1)
            @cuda threads=256 blocks=nb kernel_M2_MF_v4_compute!(
                op.ws.C_next_m, op.ws.C,
                op.coeffs.det_A, op.coeffs.det_k, op.coeffs.num_det,
                op.coeffs.stoch_E, op.coeffs.stoch_k, op.coeffs.stoch_l, op.coeffs.num_stoch,
                Int32(op.r), Int32(op.d), n)
            @cuda threads=256 blocks=nb kernel_M2_MF_v4_writeback!(
                op.ws.C, op.ws.C_next_m, Int32(op.r), Int32(op.d), n)
        end
        @cuda threads=256 blocks=nb kernel_M2_MF_v4_extract!(
            y_buf, op.ws.C, Int32(op.p), Int32(op.r), Int32(op.d), op.ws.idx_sectionStarts)
    end
    return CUDA.instantiate(g)
end

# ============================================================
# V5 cooperative kernel — ONE grid sync per step (vs 2 in v3).
# The circular buffer has r+2 slots: the new block's slot c5(n+1) is never
# read while computing step n (c5(n+1)==c5(n−k) would need k ≡ r+1 mod r+2,
# impossible for delay offsets k ≤ r), so results are written directly —
# no staging buffer, no writeback phase. At large p the cooperative path is
# grid-sync-bound, so this nearly halves the kernel time.
# ============================================================

function kernel_M2_MF_v5!(
    m_out, m_in,
    C,                                  # d×d×(r+2)×(r+2)
    det_A, det_k, num_det,
    stoch_E, stoch_k, stoch_l, num_stoch,
    p::Int32, r::Int32, d::Int32,
    idx_sectionStarts
)
    @inbounds begin
        tid    = Int32(threadIdx().x) + Int32((blockIdx().x - Int32(1)) * blockDim().x)
        stride = Int32(blockDim().x) * Int32(gridDim().x)
        rp1    = r + Int32(1)
        rp2    = r + Int32(2)
        L      = d * d
        total_C     = rp1 * rp1 * L
        total_cross = rp1 * L

        grid = _cg_this_grid()

        # init: logical block i (0..r) ↔ slot gpu_c_idx_v2(-i, rp2)
        for flat in (tid - Int32(1)):stride:(total_C - Int32(1))
            idx_j  = flat ÷ (rp1 * L)
            rem1   = flat - idx_j * (rp1 * L)
            idx_i  = rem1 ÷ L
            rem2   = rem1 - idx_i * L
            col    = rem2 ÷ d + Int32(1)
            row    = rem2 - (col - Int32(1)) * d + Int32(1)
            phys_i = gpu_c_idx_v2(-idx_i, rp2)
            phys_j = gpu_c_idx_v2(-idx_j, rp2)
            vi = idx_i * d + row
            vj = idx_j * d + col
            min_v  = vi < vj ? vi : vj
            max_v  = vi > vj ? vi : vj
            diff   = max_v - min_v
            C[row, col, phys_i, phys_j] = m_in[idx_sectionStarts[diff + Int32(1)] + min_v]
        end

        _cg_sync(grid)

        for n in Int32(0):p - Int32(1)
            next_n    = n + Int32(1)
            next_phys = gpu_c_idx_v2(next_n, rp2)
            nd = num_det[next_n]
            ns = num_stoch[next_n]

            # cross terms — direct write (slot next_phys is not read this step)
            for flat in (tid - Int32(1)):stride:(total_cross - Int32(1))
                m_offset   = flat ÷ L
                rem        = flat - m_offset * L
                col        = rem ÷ d + Int32(1)
                row        = rem - (col - Int32(1)) * d + Int32(1)
                idx_m_phys = gpu_c_idx_v2((n - r) + m_offset, rp2)
                val = 0.0
                for i in Int32(1):nd
                    idx_nk = gpu_c_idx_v2(n - det_k[i, next_n], rp2)
                    for s in Int32(1):d
                        val += det_A[row, s, i, next_n] * C[s, col, idx_nk, idx_m_phys]
                    end
                end
                C[row, col, next_phys, idx_m_phys] = val
                C[col, row, idx_m_phys, next_phys] = val
            end

            # diagonal — direct write
            for flat in (tid - Int32(1)):stride:(L - Int32(1))
                col = flat ÷ d + Int32(1)
                row = flat - (col - Int32(1)) * d + Int32(1)
                lin = row + (col - Int32(1)) * d
                val = 0.0
                for i in Int32(1):nd
                    idx_nk = gpu_c_idx_v2(n - det_k[i, next_n], rp2)
                    for j in Int32(1):nd
                        idx_nl = gpu_c_idx_v2(n - det_k[j, next_n], rp2)
                        for s in Int32(1):d
                            for t in Int32(1):d
                                val += det_A[row, s, i, next_n] * C[s, t, idx_nk, idx_nl] * det_A[col, t, j, next_n]
                            end
                        end
                    end
                end
                for i in Int32(1):ns
                    idx_nk = gpu_c_idx_v2(n - stoch_k[i, next_n], rp2)
                    idx_nl = gpu_c_idx_v2(n - stoch_l[i, next_n], rp2)
                    for s in Int32(1):d
                        for t in Int32(1):d
                            val += stoch_E[lin, s + (t - Int32(1)) * d, i, next_n] * C[s, t, idx_nk, idx_nl]
                        end
                    end
                end
                C[row, col, next_phys, next_phys] = val
            end

            _cg_sync(grid)
        end

        # extract
        for flat in (tid - Int32(1)):stride:(total_C - Int32(1))
            idx_j  = flat ÷ (rp1 * L)
            rem1   = flat - idx_j * (rp1 * L)
            idx_i  = rem1 ÷ L
            rem2   = rem1 - idx_i * L
            col    = rem2 ÷ d + Int32(1)
            row    = rem2 - (col - Int32(1)) * d + Int32(1)
            vi = idx_i * d + row
            vj = idx_j * d + col
            if vi <= vj
                val  = C[row, col, gpu_c_idx_v2(p - idx_i, rp2), gpu_c_idx_v2(p - idx_j, rp2)]
                diff = vj - vi
                m_out[idx_sectionStarts[diff + Int32(1)] + vi] = val
            end
        end
    end
    return nothing
end

# Balanced-diagonal variant for larger d: the diagonal block has only d²
# elements but O((nd²+ns)·d²) work each — at d ≥ ~6 this serializes a handful
# of threads while the rest idle. v5b splits the diagonal into
# L·(nd²+ns) partial-sum items (phase 1, scratch) + an L-item reduction
# (phase 2). Costs one extra grid sync per step (~4-8%), wins the imbalance.
function kernel_M2_MF_v5b!(
    m_out, m_in,
    C, diag_scratch,                    # scratch: L × maxterms
    det_A, det_k, num_det,
    stoch_E, stoch_k, stoch_l, num_stoch,
    p::Int32, r::Int32, d::Int32,
    idx_sectionStarts
)
    @inbounds begin
        tid    = Int32(threadIdx().x) + Int32((blockIdx().x - Int32(1)) * blockDim().x)
        stride = Int32(blockDim().x) * Int32(gridDim().x)
        rp1    = r + Int32(1)
        rp2    = r + Int32(2)
        L      = d * d
        total_C     = rp1 * rp1 * L
        total_cross = rp1 * L

        grid = _cg_this_grid()

        for flat in (tid - Int32(1)):stride:(total_C - Int32(1))
            idx_j  = flat ÷ (rp1 * L)
            rem1   = flat - idx_j * (rp1 * L)
            idx_i  = rem1 ÷ L
            rem2   = rem1 - idx_i * L
            col    = rem2 ÷ d + Int32(1)
            row    = rem2 - (col - Int32(1)) * d + Int32(1)
            phys_i = gpu_c_idx_v2(-idx_i, rp2)
            phys_j = gpu_c_idx_v2(-idx_j, rp2)
            vi = idx_i * d + row
            vj = idx_j * d + col
            min_v  = vi < vj ? vi : vj
            max_v  = vi > vj ? vi : vj
            diff   = max_v - min_v
            C[row, col, phys_i, phys_j] = m_in[idx_sectionStarts[diff + Int32(1)] + min_v]
        end

        _cg_sync(grid)

        for n in Int32(0):p - Int32(1)
            next_n    = n + Int32(1)
            next_phys = gpu_c_idx_v2(next_n, rp2)
            nd = num_det[next_n]
            ns = num_stoch[next_n]
            nterms = nd*nd + ns

            # phase 1a: cross terms (direct write, as v5)
            for flat in (tid - Int32(1)):stride:(total_cross - Int32(1))
                m_offset   = flat ÷ L
                rem        = flat - m_offset * L
                col        = rem ÷ d + Int32(1)
                row        = rem - (col - Int32(1)) * d + Int32(1)
                idx_m_phys = gpu_c_idx_v2((n - r) + m_offset, rp2)
                val = 0.0
                for i in Int32(1):nd
                    idx_nk = gpu_c_idx_v2(n - det_k[i, next_n], rp2)
                    for s in Int32(1):d
                        val += det_A[row, s, i, next_n] * C[s, col, idx_nk, idx_m_phys]
                    end
                end
                C[row, col, next_phys, idx_m_phys] = val
                C[col, row, idx_m_phys, next_phys] = val
            end

            # phase 1b: diagonal PARTIALS — one item per (element, term)
            for flat in (tid - Int32(1)):stride:(L*nterms - Int32(1))
                term = flat ÷ L + Int32(1)
                lin  = flat - (term - Int32(1))*L + Int32(1)
                col  = (lin - Int32(1)) ÷ d + Int32(1)
                row  = lin - (col - Int32(1)) * d
                val = 0.0
                if term <= nd*nd
                    i = (term - Int32(1)) ÷ nd + Int32(1)
                    j = term - (i - Int32(1))*nd
                    idx_nk = gpu_c_idx_v2(n - det_k[i, next_n], rp2)
                    idx_nl = gpu_c_idx_v2(n - det_k[j, next_n], rp2)
                    for s in Int32(1):d
                        for t in Int32(1):d
                            val += det_A[row, s, i, next_n] * C[s, t, idx_nk, idx_nl] * det_A[col, t, j, next_n]
                        end
                    end
                else
                    i = term - nd*nd
                    idx_nk = gpu_c_idx_v2(n - stoch_k[i, next_n], rp2)
                    idx_nl = gpu_c_idx_v2(n - stoch_l[i, next_n], rp2)
                    for s in Int32(1):d
                        for t in Int32(1):d
                            val += stoch_E[lin, s + (t - Int32(1)) * d, i, next_n] * C[s, t, idx_nk, idx_nl]
                        end
                    end
                end
                diag_scratch[lin, term] = val
            end

            _cg_sync(grid)

            # phase 2: reduce partials into the diagonal block
            for lin0 in (tid - Int32(1)):stride:(L - Int32(1))
                lin = lin0 + Int32(1)
                col = lin0 ÷ d + Int32(1)
                row = lin - (col - Int32(1)) * d
                val = 0.0
                for term in Int32(1):nterms
                    val += diag_scratch[lin, term]
                end
                C[row, col, next_phys, next_phys] = val
            end

            _cg_sync(grid)
        end

        for flat in (tid - Int32(1)):stride:(total_C - Int32(1))
            idx_j  = flat ÷ (rp1 * L)
            rem1   = flat - idx_j * (rp1 * L)
            idx_i  = rem1 ÷ L
            rem2   = rem1 - idx_i * L
            col    = rem2 ÷ d + Int32(1)
            row    = rem2 - (col - Int32(1)) * d + Int32(1)
            vi = idx_i * d + row
            vj = idx_j * d + col
            if vi <= vj
                val  = C[row, col, gpu_c_idx_v2(p - idx_i, rp2), gpu_c_idx_v2(p - idx_j, rp2)]
                diff = vj - vi
                m_out[idx_sectionStarts[diff + Int32(1)] + vi] = val
            end
        end
    end
    return nothing
end

struct MFGPUMappingOperator_v5b <: AbstractMatrix{Float64}
    coeffs::MFGPUCoefficients
    D::Int; r::Int; d::Int; p::Int; n_sm::Int
    C::CuArray{Float64,4}
    diag_scratch::CuArray{Float64,2}
    idx_sectionStarts::CuArray{Int32,1}
end
Base.size(op::MFGPUMappingOperator_v5b) = (op.D, op.D)
Base.size(op::MFGPUMappingOperator_v5b, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFGPUMappingOperator_v5b) = Float64
LinearAlgebra.issymmetric(op::MFGPUMappingOperator_v5b) = false
LinearAlgebra.ishermitian(op::MFGPUMappingOperator_v5b) = false

function LinearAlgebra.mul!(y::AbstractVector, op::MFGPUMappingOperator_v5b, x::AbstractVector)
    @cuda threads=256 blocks=op.n_sm cooperative=true kernel_M2_MF_v5b!(
        y, x, op.C, op.diag_scratch,
        op.coeffs.det_A, op.coeffs.det_k, op.coeffs.num_det,
        op.coeffs.stoch_E, op.coeffs.stoch_k, op.coeffs.stoch_l, op.coeffs.num_stoch,
        Int32(op.p), Int32(op.r), Int32(op.d), op.idx_sectionStarts
    )
    return y
end

struct MFGPUMappingOperator_v5 <: AbstractMatrix{Float64}
    coeffs::MFGPUCoefficients
    D::Int; r::Int; d::Int; p::Int; n_sm::Int
    C::CuArray{Float64,4}                       # (d,d,r+2,r+2)
    idx_sectionStarts::CuArray{Int32,1}
end
Base.size(op::MFGPUMappingOperator_v5) = (op.D, op.D)
Base.size(op::MFGPUMappingOperator_v5, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFGPUMappingOperator_v5) = Float64
LinearAlgebra.issymmetric(op::MFGPUMappingOperator_v5) = false
LinearAlgebra.ishermitian(op::MFGPUMappingOperator_v5) = false

function LinearAlgebra.mul!(y::AbstractVector, op::MFGPUMappingOperator_v5, x::AbstractVector)
    @cuda threads=256 blocks=op.n_sm cooperative=true kernel_M2_MF_v5!(
        y, x, op.C,
        op.coeffs.det_A, op.coeffs.det_k, op.coeffs.num_det,
        op.coeffs.stoch_E, op.coeffs.stoch_k, op.coeffs.stoch_l, op.coeffs.num_stoch,
        Int32(op.p), Int32(op.r), Int32(op.d), op.idx_sectionStarts
    )
    return y
end

# ============================================================
# User API
#
# Zero-Sync policy: coefficients are uploaded to the GPU once, the whole
# Krylov iteration runs with device-resident vectors (each matvec is a
# single cooperative kernel launch, or one CUDA-graph replay on devices
# without cooperative-launch support), and only scalars — the Floquet
# multiplier / convergence norms — travel back to the host.
# ============================================================

_gpu_coop_supported() =
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) == 1

_gpu_n_sm() = Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))

# krylovdim: KrylovKit's default (30) unless the Krylov basis would not fit
# in GPU memory. (An earlier D÷500 heuristic forced tiny subspaces on small
# problems, causing many restarts — measured 2× slower; do not reintroduce.)
function _gpu_krylovdim(D::Int, krylovdim::Int)
    krylovdim > 0 && return krylovdim
    avail    = Int(CUDA.free_memory())
    reserved = D * 8 + 64 * 1024^2   # state vector + 64 MB headroom
    mem_kd   = max(4, (avail - reserved) ÷ max(1, D * 8))
    return min(30, mem_kd)
end

# The homogeneous second-moment mapping as a device-resident linear operator:
# cooperative single-launch kernel when supported (v5: one grid sync/step),
# CUDA-graph replay otherwise.
function _make_m2_gpu_operator(gpu_coeffs::MFGPUCoefficients, D, r, d, p, ws)
    n_sm = _gpu_n_sm()
    if _gpu_coop_supported()
        C5 = CUDA.zeros(Float64, d, d, r+2, r+2)
        # Right-size the cooperative grid: the per-step parallel width is
        # (r+1)·d² elements; extra blocks only add grid-sync latency (matters
        # on large-SM devices like A100 where n_sm ≫ needed blocks).
        nblk = clamp(cld((r+1)*d*d, 256), 1, n_sm)
        if d >= 6
            # balanced-diagonal kernel: avoids the d² -element serial diagonal
            # phase that starves the GPU at larger state dimensions
            mt = size(gpu_coeffs.det_A, 3)^2 + size(gpu_coeffs.stoch_E, 3)
            scratch = CUDA.zeros(Float64, d*d, mt)
            return MFGPUMappingOperator_v5b(gpu_coeffs, D, r, d, p, nblk, C5,
                                            scratch, ws.idx_sectionStarts)
        end
        return MFGPUMappingOperator_v5(gpu_coeffs, D, r, d, p, nblk, C5, ws.idx_sectionStarts)
    else
        x_buf = CUDA.zeros(Float64, D)
        y_buf = CUDA.zeros(Float64, D)
        op_v4 = MFGPUMappingOperator_v4(gpu_coeffs, D, r, d, p, n_sm, ws)
        # Pre-warm the v4 kernels (JIT inside graph capture is not allowed)
        mul!(y_buf, op_v4, x_buf)
        CUDA.synchronize()
        CUDA.fill!(ws.C, 0.0)
        exec = _build_v4_graph(op_v4, x_buf, y_buf)
        return MFGPUMappingOperator_v4g(gpu_coeffs, D, r, d, p, n_sm, ws, x_buf, y_buf, exec)
    end
end

function spectralRadiusOfMapping_GPU(dm::stDiscreteMapping_M2_MF;
                                     krylovdim::Int=0, args...)
    d = size(dm.coeffs.det[1][1][1], 1)
    p = dm.rst.n_steps
    r = div(dm.rst.n, d) - 1
    D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]

    gpu_coeffs = extract_gpu_coeffs(dm.coeffs, p, d, false)
    ws = MFGPUWorkspace(dm.rst)
    op = _make_m2_gpu_operator(gpu_coeffs, D, r, d, p, ws)
    CUDA.seed!(42)
    C0 = CUDA.rand(Float64, D)

    vals, _, _ = eigsolve(op, C0, 1, :LM; ishermitian=false,
                          krylovdim=_gpu_krylovdim(D, krylovdim), args...)
    return abs(vals[1])
end

function fixPointOfMapping_GPU(dm::stDiscreteMapping_M2_MF; rtol=1e-15, args...)
    d = size(dm.coeffs.det[1][1][1], 1)
    p = dm.rst.n_steps
    r = div(dm.rst.n, d) - 1

    D1 = (r+1)*d
    D2 = StochasticSemiDiscretizationMethod.CovVecIdx(D1).sectionStarts[end]

    gpu_coeffs = extract_gpu_coeffs(dm.coeffs, p, d, true)
    ws = MFGPUWorkspace(dm.rst)

    op1 = M1MFGPUMappingOperator(gpu_coeffs, D1, r, d, p, ws)
    k1  = CUDA.zeros(Float64, D1)
    @cuda threads=1024 blocks=1 kernel_M1_MF!(
        k1, CUDA.zeros(Float64, D1), ws.v, ws.v_next,
        gpu_coeffs.det_A, gpu_coeffs.det_k, gpu_coeffs.num_det, gpu_coeffs.detV,
        Int32(p), Int32(r), Int32(d), true
    )
    v_star, _ = linsolve(MFGPUIMinusPhiOperator(op1), k1; rtol=rtol, args...)

    # Affine constant k2 = Φ(0) with additive terms — needs the additive-capable
    # (v1) kernel; the linsolve iterations only need the homogeneous mapping,
    # which runs on the fast operator.
    k2 = CUDA.zeros(Float64, D2)
    @cuda threads=256 blocks=1 kernel_M2_MF!(
        k2, ws.v_in_zero, CUDA.zeros(Float64, D2), v_star,
        ws.C, ws.v, ws.C_next_m, ws.v_next,
        gpu_coeffs.det_A, gpu_coeffs.det_k, gpu_coeffs.num_det, gpu_coeffs.detV,
        gpu_coeffs.stoch_E, gpu_coeffs.stoch_k, gpu_coeffs.stoch_l, gpu_coeffs.num_stoch,
        gpu_coeffs.stochV, gpu_coeffs.stochGV, gpu_coeffs.stochGV_k, gpu_coeffs.num_stochGV,
        Int32(p), Int32(r), Int32(d), true, ws.idx_sectionStarts
    )
    op2 = _make_m2_gpu_operator(gpu_coeffs, D2, r, d, p, ws)
    m_star, _ = linsolve(MFGPUIMinusPhiOperator(op2), k2; rtol=rtol, args...)
    return m_star
end

# Automatic dispatch: CPU-MF below the measured CPU/GPU crossover, GPU above.
# Crossover measured 2026-07-02 on a Quadro P4000 vs 1-thread CPU MF
# (stoch. Mathieu d=2, q=2): GPU wins from D ≈ 10⁴ (p ≈ 140) and is ~2.3×
# faster at D ≈ 2·10⁵ (p = 640).
function spectralRadiusOfMapping_auto(dm::stDiscreteMapping_M2_MF;
                                       cpu_threshold::Int = 10_000,
                                       krylovdim::Int     = 0,
                                       args...)
    d = size(dm.coeffs.det[1][1][1], 1)
    r = div(dm.rst.n, d) - 1
    D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
    if D < cpu_threshold || !CUDA.functional()
        return spectralRadiusOfMapping_MF(dm; args...)
    else
        return spectralRadiusOfMapping_GPU(dm; krylovdim=krylovdim, args...)
    end
end
end # module StochasticSemiDiscretizationMethodCUDAExt
