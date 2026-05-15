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
function apply_mapping_M2_MF!(ws::MFWorkspace{d, L}, rst::AbstractResult{d}, coeffs::MFCoefficients{d, L, L2, Ld}, m_in::AbstractVector, v_in::AbstractVector; include_additive=false) where {d, L, L2, Ld}
    r = div(rst.n, d) - 1
    p = rst.n_steps
    idx = StochasticSemiDiscretizationMethod.CovVecIdx((r + 1) * d)
    
    C = ws.C
    v = ws.v
    C_next_m = ws.C_next_m
    c_idx(n) = mod(n, r+1) + 1
    
    # 0. Initialize
    for i in 0:r
        v[c_idx(-i)] = SVector{d, Float64}(v_in[i*d+1:(i+1)*d])
        for j in 0:r
            Mat = zeros(MMatrix{d, d, Float64, L})
            for r_row in 1:d, r_col in 1:d
                Mat[r_row, r_col] = m_in[idx(i*d + r_row, j*d + r_col)]
            end
            C[c_idx(-i), c_idx(-j)] = SMatrix{d,d,Float64,L}(Mat)
        end
    end
    
    # 1. Propagate
    for n in 0:p-1
        next_n = n + 1
        det_step = coeffs.det[next_n]
        stoch_op = coeffs.stoch_op[next_n]
        detV = coeffs.detV[next_n]
        
        # v(n+1)
        v_next_M = zeros(MVector{d, Float64})
        for (A, k) in det_step
            v_next_M .+= A * v[c_idx(n-k)]
        end
        if include_additive
            v_next_M .+= detV
        end
        v_next_S = SVector{d, Float64}(v_next_M)
        
        # C(n+1, m)
        for (i, m) in enumerate(n-r:n)
            res = zeros(MMatrix{d, d, Float64, L})
            for (A, k) in det_step
                res .+= A * C[c_idx(n-k), c_idx(m)]
            end
            if include_additive
                res .+= detV * v[c_idx(m)]'
            end
            C_next_m[i] = SMatrix{d,d,Float64,L}(res)
        end
        
        # C(n+1, n+1)
        res_diag = zeros(MMatrix{d, d, Float64, L})
        for (Ak, k) in det_step, (Al, l) in det_step
            res_diag .+= Ak * C[c_idx(n-k), c_idx(n-l)] * Al'
        end
        
        for (E_mat, k, l) in stoch_op
            Ckl_vec = SVector{L, Float64}(C[c_idx(n-k), c_idx(n-l)])
            res_vec = E_mat * Ckl_vec
            res_diag .+= SMatrix{d, d, Float64, L}(res_vec)
        end
        
        if include_additive
            Fv = v_next_S - detV
            res_diag .+= detV * Fv' + Fv * detV' + detV * detV'
            res_diag .+= coeffs.stochV[next_n]
            for (E_Gg_mat, k, w) in coeffs.stochGV[next_n]
                term_vec = E_Gg_mat * v[c_idx(n-k)]
                term = SMatrix{d, d, Float64, L}(term_vec)
                res_diag .+= term + term'
            end
        end
        
        C_final_diag = SMatrix{d,d,Float64,L}(res_diag)
        
        # Update
        v[c_idx(next_n)] = v_next_S
        for (i, m) in enumerate(n-r:n)
            C[c_idx(next_n), c_idx(m)] = C_next_m[i]
            C[c_idx(m), c_idx(next_n)] = C_next_m[i]'
        end
        C[c_idx(next_n), c_idx(next_n)] = C_final_diag
    end
    
    # 2. Extract
    m_out = zeros(length(m_in))
    for i in 0:r, j in 0:r
        p_i_idx = c_idx(p-i)
        p_j_idx = c_idx(p-j)
        Mat = C[p_i_idx, p_j_idx]
        for r_row in 1:d, r_col in 1:d
            vi = i*d + r_row
            vj = j*d + r_col
            if vi <= vj
                m_out[idx(vi, vj)] = Mat[r_row, r_col]
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
