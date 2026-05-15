using SparseArrays
using LinearAlgebra
using StaticArrays
using Arpack
using IterativeSolvers

# Multiplication-Free Stochastic Semi-Discretization Method

struct MFCoefficients{d, T, L}
    det::Vector{Vector{Tuple{SMatrix{d, d, Float64, L}, Int}}}
    stoch::Vector{Vector{Tuple{SMatrix{d, d, SVector{T, Float64}, L}, Int, Int}}}
    # Additive terms
    detV::Vector{SVector{d, Float64}}
    stochV::Vector{SMatrix{d, d, Float64, L}} # E[g_n g_n']
    stochGV::Vector{Vector{Tuple{Array{Float64, 3}, Int, Int}}} # E[G_n,k(a,c) * g_n(b)] -> (a,b,c)
end

struct stDiscreteMapping_M2_MF{tT, mx1T, mx12T, v1T, v2T, rstT, coeffT}
    ts::Vector{tT}
    M1_MXs::Vector{mx1T} # F [time]
    M1_Vs::Vector{v1T} # c_1 [time]
    M1toM2_MXs::Vector{mx12T} # C [time]
    M2_Vs::Vector{v2T} # c_2 [time]
    rst::rstT
    coeffs::coeffT
end

function get_all_coefficients(rst::AbstractResult{d}) where d
    p = rst.n_steps
    L = d*d
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
    
    K = length(rst.itoisometrymethod)
    stoch_all = [Tuple{SMatrix{d, d, SVector{K, Float64}, L}, Int, Int}[] for _ in 1:p]
    for (noise_type, stsubmxs) in enumerate(rst.stsubMXs)
        for i in 1:p
            stsmx = stsubmxs[i]
            w = stsmx.nID
            for (range_idx, (r_target, r_source)) in enumerate(stsmx.ranges)
                k = Int((r_source.start - 1) / d)
                G = stsmx.MXfun[range_idx] 
                push!(stoch_all[i], (SMatrix{d,d,SVector{K,Float64},L}(G), k, w))
            end
        end
    end

    detV = [SVector{d, Float64}(zeros(d)) for _ in 1:p]
    if rst.calculate_additive && !isempty(rst.subVs)
        for i in 1:p
            detV[i] = SVector{d, Float64}(rst.subVs[i].V)
        end
    end

    stochV = [zeros(SMatrix{d, d, Float64, L}) for _ in 1:p]
    stochGV = [Tuple{Array{Float64, 3}, Int, Int}[] for _ in 1:p]
    
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
                            E_Gg = zeros(d, d, d) # (a, b, c) -> E[G_ac * g_b]
                            for a in 1:d, b in 1:d, c in 1:d
                                E_Gg[a, b, c] = rst.itoisometrymethod(Gfun[a, c], gfun[b])
                            end
                            push!(stochGV[i], (E_Gg, k, w))
                        end
                    end
                end
            end
            stochV[i] = SMatrix{d,d,Float64,L}(res_noise)
        end
    end

    return MFCoefficients{d, K, L}(det_all, stoch_all, detV, stochV, stochGV)
end

function DiscreteMapping_M2_MF(rst::AbstractResult)
    dm1 = DiscreteMapping_M1(rst)
    coeffs = get_all_coefficients(rst)
    M1toM2_MXs = typeof(dm1.M1_MXs)()
    M2_Vs = typeof(dm1.M1_Vs)()
    return stDiscreteMapping_M2_MF(dm1.ts, dm1.M1_MXs, dm1.M1_Vs, M1toM2_MXs, M2_Vs, rst, coeffs)
end

function apply_stoch_mapping(Gk::SMatrix{d,d,SVector{K,Float64},L}, Gl::SMatrix{d,d,SVector{K,Float64},L}, Ckl, rst) where {d, K, L}
    res = zeros(MMatrix{d, d, Float64, L})
    for a in 1:d, b in 1:d
        val = 0.0
        for c in 1:d, dd in 1:d
            val += rst.itoisometrymethod(Gk[a,c], Gl[b,dd]) * Ckl[c,dd]
        end
        res[a, b] = val
    end
    return SMatrix{d,d,Float64,L}(res)
end

function apply_mapping_M1_MF(rst::AbstractResult{d}, coeffs::MFCoefficients{d, K, L}, v_in::AbstractVector; include_additive=false) where {d, K, L}
    r = StochasticSemiDiscretizationMethod.rOfDelay(rst.ts[end], rst.method)
    p = rst.n_steps
    v = [zero(SVector{d, Float64}) for i in 1:(r+1)]
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

function apply_mapping_M2_MF(rst::AbstractResult{d}, coeffs::MFCoefficients{d, K, L}, m_in::AbstractVector, v_in::AbstractVector; include_additive=false) where {d, K, L}
    r = StochasticSemiDiscretizationMethod.rOfDelay(rst.ts[end], rst.method)
    p = rst.n_steps
    idx = StochasticSemiDiscretizationMethod.CovVecIdx((r + 1) * d)
    
    C = [zero(SMatrix{d, d, Float64, L}) for i in 1:(r+1), j in 1:(r+1)]
    v = [zero(SVector{d, Float64}) for i in 1:(r+1)]
    c_idx(n) = mod(n, r+1) + 1
    
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
    
    for n in 0:p-1
        next_n = n + 1
        det_step = coeffs.det[next_n]
        stoch_step = coeffs.stoch[next_n]
        detV = coeffs.detV[next_n]
        
        # 1. Calculate next v
        v_next = zeros(MVector{d, Float64})
        for (A, k) in det_step
            v_next .+= A * v[c_idx(n-k)]
        end
        if include_additive
            v_next .+= detV
        end
        v_next_S = SVector{d, Float64}(v_next)
        
        # 2. Calculate next C(next_n, m)
        C_next_m = Vector{SMatrix{d, d, Float64, L}}(undef, r + 1)
        for (i, m) in enumerate(next_n-r:n)
            res = zeros(MMatrix{d, d, Float64, L})
            for (A, k) in det_step
                res .+= A * C[c_idx(n-k), c_idx(m)]
            end
            if include_additive
                res .+= detV * v[c_idx(m)]'
            end
            C_next_m[i] = SMatrix{d,d,Float64,L}(res)
        end
        
        # 3. Calculate next C(next_n, next_n)
        res_diag = zeros(MMatrix{d, d, Float64, L})
        for (Ak, k) in det_step, (Al, l) in det_step
            res_diag .+= Ak * C[c_idx(n-k), c_idx(n-l)] * Al'
        end
        for (Gk, k, wk) in stoch_step, (Gl, l, wl) in stoch_step
            if wk == wl
                res_diag .+= apply_stoch_mapping(Gk, Gl, C[c_idx(n-k), c_idx(n-l)], rst)
            end
        end
        
        if include_additive
            # Deterministic additive terms
            # Fv f' + f (Fv)' + f f'
            # Fv = sum Ak v(n-k)
            Fv = zeros(MVector{d, Float64})
            for (Ak, k) in det_step
                Fv .+= Ak * v[c_idx(n-k)]
            end
            res_diag .+= Fv * detV' .+ detV * Fv' .+ detV * detV'
            
            # Stochastic additive terms E[g g']
            res_diag .+= coeffs.stochV[next_n]
            
            # Cross terms E[G y g' + g y' G']
            for (E_Gg, k, w) in coeffs.stochGV[next_n]
                # sum_c E[G_ac g_b] y_c
                term = zeros(MMatrix{d, d, Float64, L})
                for a in 1:d, b in 1:d, c in 1:d
                    term[a, b] += E_Gg[a, b, c] * v[c_idx(n-k)][c]
                end
                res_diag .+= term .+ term'
            end
        end
        
        C_next_m[r + 1] = SMatrix{d,d,Float64,L}(res_diag)
        
        # Update buffer
        v[c_idx(next_n)] = v_next_S
        for (i, m) in enumerate(next_n-r:next_n)
            C[c_idx(next_n), c_idx(m)] = C_next_m[i]
            C[c_idx(m), c_idx(next_n)] = C_next_m[i]'
        end
    end
    
    m_out = zeros(length(m_in))
    for i in 0:r, j in 0:r
        Mat = C[c_idx(p-i), c_idx(p-j)]
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

function apply_mapping_M2_MF(rst::AbstractResult{d}, coeffs::MFCoefficients{d, K, L}, m_in::AbstractVector; include_additive=false) where {d, K, L}
    r = StochasticSemiDiscretizationMethod.rOfDelay(rst.ts[end], rst.method)
    v_in = zeros((r+1)*d) # Homogeneous if no v_in provided
    return apply_mapping_M2_MF(rst, coeffs, m_in, v_in, include_additive=include_additive)
end

struct MFMappingOperator{ResultT, CoeffT} <: AbstractMatrix{Float64}
    rst::ResultT
    coeffs::CoeffT
    D::Int
end
Base.size(op::MFMappingOperator) = (op.D, op.D)
Base.size(op::MFMappingOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFMappingOperator) = Float64
LinearAlgebra.issymmetric(op::MFMappingOperator) = false
LinearAlgebra.ishermitian(op::MFMappingOperator) = false

function LinearAlgebra.mul!(y::AbstractVector, op::MFMappingOperator, x::AbstractVector)
    y .= apply_mapping_M2_MF(op.rst, op.coeffs, x, include_additive=false)
end

struct M1MFMappingOperator{ResultT, CoeffT} <: AbstractMatrix{Float64}
    rst::ResultT
    coeffs::CoeffT
    D::Int
end
Base.size(op::M1MFMappingOperator) = (op.D, op.D)
Base.size(op::M1MFMappingOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::M1MFMappingOperator) = Float64

function LinearAlgebra.mul!(y::AbstractVector, op::M1MFMappingOperator, x::AbstractVector)
    y .= apply_mapping_M1_MF(op.rst, op.coeffs, x, include_additive=false)
end

function spectralRadiusOfMapping_MF(dm::stDiscreteMapping_M2_MF{tT, mx1T, mx12T, v1T, v2T, ResultT, coeffT}; args...) where {tT, mx1T, mx12T, v1T, v2T, d, ResultT <: AbstractResult{d}, coeffT}
    r = StochasticSemiDiscretizationMethod.rOfDelay(dm.ts[end], dm.rst.method)
    D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
    op = MFMappingOperator(dm.rst, dm.coeffs, D)
    res, _ = eigs(op, v0=rand(D); args...)
    return abs(res[1])
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

function fixPointOfMapping_MF(dm::stDiscreteMapping_M2_MF{tT, mx1T, mx12T, v1T, v2T, ResultT, coeffT}; args...) where {tT, mx1T, mx12T, v1T, v2T, d, ResultT <: AbstractResult{d}, coeffT}
    r = StochasticSemiDiscretizationMethod.rOfDelay(dm.ts[end], dm.rst.method)
    D1 = (r+1)*d
    D2 = StochasticSemiDiscretizationMethod.CovVecIdx(D1).sectionStarts[end]
    
    # 1. First moment fixed point
    op1 = M1MFMappingOperator(dm.rst, dm.coeffs, D1)
    k1 = apply_mapping_M1_MF(dm.rst, dm.coeffs, zeros(D1), include_additive=true)
    v_star = gmres(IMinusPhiOperator(op1), k1; reltol=1e-15, args...)
    
    # 2. Second moment additive part
    k2 = apply_mapping_M2_MF(dm.rst, dm.coeffs, zeros(D2), v_star, include_additive=true)
    
    # 3. Second moment fixed point
    op2 = MFMappingOperator(dm.rst, dm.coeffs, D2)
    m_star = gmres(IMinusPhiOperator(op2), k2; reltol=1e-15, args...)
    return m_star
end

