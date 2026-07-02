# =============================================================================
# Factored-operator second-moment engine.
#
# The dense MF path stores, per step and per stochastic (k,l) pair, a d²×d²
# matrix `stoch_op` = E[G_l ⊗ G_k] (d⁴ entries) and applies it as a dense
# matvec on vec(C). That d⁴ storage is the wall that stops the method at
# d ≈ 10 (StaticArrays compile blow-up too).
#
# The Itô isometry is  E[G_k⊗G_l][…] = Σₘ (iim[m]·dt)·G_k[…]⁽ᵐ⁾·G_l[…]⁽ᵐ⁾,
# i.e. a sum of K rank-structured terms. Scaling each sample by √(iim[m]·dt)
# turns the second-moment stochastic step into
#       C ↦ Σ_pairs Σₘ  G̃_k⁽ᵐ⁾ · C · (G̃_l⁽ᵐ⁾)ᵀ
# — K products of d×d, O(K·d³) work and O(K·d²) storage (no d⁴ anywhere).
# This is mathematically the SAME operator, so ρ(H) and the fixpoint are
# numerically identical to the dense path (agree to ~1e-13, summation-order
# rounding only). Everything here uses plain Matrix{Float64}, so it scales to
# arbitrary d (10-DoF → d=20, 100-DoF → d=200).
#
# Public: spectralRadiusOfMapping_MF_factored, fixPointOfMapping_MF_factored.
# =============================================================================

struct MFFactoredCoefficients
    d::Int
    K::Int
    # per step: (A_k :: d×d, lag k)
    det::Vector{Vector{Tuple{Matrix{Float64}, Int}}}
    # per step: (scaled samples G̃_k[m] :: Vector of K d×d, lag k, noise id w)
    stoch::Vector{Vector{Tuple{Vector{Matrix{Float64}}, Int, Int}}}
    # additive (for fixpoint); empty vectors when not requested
    detV::Vector{Vector{Float64}}
    stochV::Vector{Matrix{Float64}}
    # cross state×noise: (E_Gg :: Vector of K (d×d) scaled samples paired with g̃, lag k, w)
    #   stored as (Gsamples::Vector{Matrix}, gsamples::Vector{Vector{Float64}}, k, w)
    stochGV::Vector{Vector{Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}, Int, Int}}}
end

# Extract factored coefficients directly from the Result (mirrors
# get_all_coefficients but keeps stochastic terms in K-sample factored form
# and uses plain Matrix throughout).
function get_factored_coefficients(rst::AbstractResult{d}; include_additive::Bool=false) where d
    p = rst.n_steps
    iim = rst.itoisometrymethod
    K = length(iim)
    # quadrature weight per sample, folded as √ into each factor
    sqw = [sqrt(iim[m] * iim.dt) for m in 1:K]

    # deterministic
    det = [Tuple{Matrix{Float64}, Int}[] for _ in 1:p]
    for submxs in rst.subMXs
        for i in 1:p
            smx = submxs[i]
            for (range_idx, (r_target, r_source)) in enumerate(smx.ranges)
                k = Int((r_source.start - 1) / d)
                push!(det[i], (Matrix{Float64}(smx.MXs[range_idx]), k))
            end
        end
    end

    # stochastic: samples G_k[m] scaled by √weight
    stoch = [Tuple{Vector{Matrix{Float64}}, Int, Int}[] for _ in 1:p]
    for stsubmxs in rst.stsubMXs
        for i in 1:p
            stsmx = stsubmxs[i]
            w = stsmx.nID
            for (range_idx, (r_target, r_source)) in enumerate(stsmx.ranges)
                k = Int((r_source.start - 1) / d)
                G = stsmx.MXfun[range_idx]                  # d×d of SVector{K}
                samples = [Matrix{Float64}(undef, d, d) for _ in 1:K]
                for m in 1:K, a in 1:d, c in 1:d
                    samples[m][a, c] = sqw[m] * G[a, c][m]
                end
                push!(stoch[i], (samples, k, w))
            end
        end
    end

    detV     = Vector{Float64}[]
    stochV   = Matrix{Float64}[]
    stochGV  = [Tuple{Vector{Matrix{Float64}}, Vector{Vector{Float64}}, Int, Int}[] for _ in 1:p]
    if include_additive
        detV = [zeros(d) for _ in 1:p]
        if rst.calculate_additive && !isempty(rst.subVs)
            for i in 1:p; detV[i] = Vector{Float64}(rst.subVs[i].V); end
        end
        stochV = [zeros(d, d) for _ in 1:p]
        if rst.calculate_additive && !isempty(rst.stsubVs)
            for i in 1:p
                res = zeros(d, d)
                for stsubv_list in rst.stsubVs
                    stsv = stsubv_list[i]
                    w = stsv.nID
                    # E[g g'] via isometry
                    gs = [ [sqw[m]*stsv.Vfun[a][m] for a in 1:d] for m in 1:K ]
                    for m in 1:K, a in 1:d, b in 1:d
                        res[a, b] += gs[m][a] * gs[m][b]
                    end
                    # cross E[G_k y_k g'] terms
                    for stsmx_list in rst.stsubMXs
                        stsmx = stsmx_list[i]
                        stsmx.nID == w || continue
                        for (range_idx, (r_target, r_source)) in enumerate(stsmx.ranges)
                            k = Int((r_source.start - 1) / d)
                            G = stsmx.MXfun[range_idx]
                            Gsamp = [Matrix{Float64}(undef, d, d) for _ in 1:K]
                            for m in 1:K, a in 1:d, c in 1:d
                                Gsamp[m][a, c] = sqw[m] * G[a, c][m]
                            end
                            push!(stochGV[i], (Gsamp, gs, k, w))
                        end
                    end
                end
                stochV[i] = res
            end
        end
    end

    return MFFactoredCoefficients(d, K, det, stoch, detV, stochV, stochGV)
end

# Workspace: plain Matrix covariance window (r+1)×(r+1) of d×d blocks.
struct MFFactoredWorkspace
    C::Matrix{Matrix{Float64}}
    v::Vector{Vector{Float64}}
    tmp::Matrix{Float64}          # d×d scratch (per-thread copies made inside)
end
function MFFactoredWorkspace(d::Int, r::Int)
    C = [zeros(d, d) for _ in 1:(r+1), _ in 1:(r+1)]
    v = [zeros(d) for _ in 1:(r+1)]
    MFFactoredWorkspace(C, v, zeros(d, d))
end

# One application of the homogeneous (or affine) second-moment map, factored.
function apply_mapping_M2_factored!(ws::MFFactoredWorkspace, cf::MFFactoredCoefficients,
                                    rst::AbstractResult, m_in::AbstractVector, v_in::AbstractVector;
                                    include_additive::Bool=false)
    d = cf.d; K = cf.K
    r = div(rst.n, d) - 1
    p = rst.n_steps
    idx = CovVecIdx((r + 1) * d)
    C = ws.C; v = ws.v
    c_idx(n) = mod(n, r + 1) + 1

    # unpack m_in → C, v_in → v
    for i in 0:r
        vi = v[c_idx(-i)]
        for q in 1:d; vi[q] = v_in[i*d + q]; end
        for j in 0:r
            Cij = C[c_idx(-i), c_idx(-j)]
            for a in 1:d, b in 1:d
                Cij[a, b] = m_in[idx(i*d + a, j*d + b)]
            end
        end
    end

    Cnext = [zeros(d, d) for _ in 0:r]           # new cross blocks C(n+1, m)
    for n in 0:p-1
        nn = n + 1
        det_step = cf.det[nn]
        stoch_step = cf.stoch[nn]

        # v(n+1)
        vnext = zeros(d)
        for (A, k) in det_step
            mul!(vnext, A, v[c_idx(n-k)], 1.0, 1.0)
        end
        if include_additive && !isempty(cf.detV)
            vnext .+= cf.detV[nn]
        end

        # cross blocks C(n+1, m) = Σ_k A_k C(n-k, m)  [+ detV v(m)' if additive]
        for (ii, mlag) in enumerate(n-r:n)
            cm = c_idx(mlag)
            res = Cnext[ii]; fill!(res, 0.0)
            for (A, k) in det_step
                mul!(res, A, C[c_idx(n-k), cm], 1.0, 1.0)
            end
            if include_additive && !isempty(cf.detV)
                vm = v[cm]
                for a in 1:d, b in 1:d; res[a, b] += cf.detV[nn][a] * vm[b]; end
            end
        end

        # diagonal C(n+1,n+1)
        diag = zeros(d, d)
        # deterministic  Σ_{k,l} A_k C(n-k,n-l) A_l'
        T = zeros(d, d)
        for (Ak, k) in det_step
            row_blk = c_idx(n-k)
            for (Al, l) in det_step
                mul!(T, Ak, C[row_blk, c_idx(n-l)])       # T = Ak C
                mul!(diag, T, transpose(Al), 1.0, 1.0)    # diag += T Al'
            end
        end
        # stochastic (factored)  Σ_{k,l:wk==wl} Σ_m G̃k^m C(n-k,n-l) G̃l^mᵀ
        for (Gk, k, wk) in stoch_step
            row_blk = c_idx(n-k)
            for (Gl, l, wl) in stoch_step
                wk == wl || continue
                col_blk = c_idx(n-l)
                Ckl = C[row_blk, col_blk]
                for m in 1:K
                    mul!(T, Gk[m], Ckl)                    # T = G̃k^m C
                    mul!(diag, T, transpose(Gl[m]), 1.0, 1.0)
                end
            end
        end
        if include_additive
            if !isempty(cf.detV)
                fv = vnext .- cf.detV[nn]
                dv = cf.detV[nn]
                for a in 1:d, b in 1:d
                    diag[a, b] += dv[a]*fv[b] + fv[a]*dv[b] + dv[a]*dv[b]
                end
            end
            if !isempty(cf.stochV)
                diag .+= cf.stochV[nn]
            end
            for (Gsamp, gs, k, w) in cf.stochGV[nn]
                vk = v[c_idx(n-k)]
                for m in 1:K
                    # term = (G̃^m v_k) g̃^mᵀ  + transpose
                    Gv = Gsamp[m] * vk
                    for a in 1:d, b in 1:d
                        diag[a, b] += Gv[a]*gs[m][b] + gs[m][a]*Gv[b]
                    end
                end
            end
        end

        # commit
        for q in 1:d; v[c_idx(nn)][q] = vnext[q]; end
        cn = c_idx(nn)
        for (ii, mlag) in enumerate(n-r:n)
            cm = c_idx(mlag)
            Cc = C[cn, cm]; Cr = C[cm, cn]
            for a in 1:d, b in 1:d
                val = Cnext[ii][a, b]
                Cc[a, b] = val; Cr[b, a] = val
            end
        end
        Cd = C[cn, cn]
        for a in 1:d, b in 1:d; Cd[a, b] = diag[a, b]; end
    end

    # extract C → m_out
    m_out = zeros(length(m_in))
    for i in 0:r, j in 0:r
        Mat = C[c_idx(p-i), c_idx(p-j)]
        for a in 1:d, b in 1:d
            vi = i*d + a; vj = j*d + b
            vi <= vj && (m_out[idx(vi, vj)] = Mat[a, b])
        end
    end
    return m_out
end

# first-moment map (for the fixpoint affine constant)
function apply_mapping_M1_factored!(ws::MFFactoredWorkspace, cf::MFFactoredCoefficients,
                                    rst::AbstractResult, v_in::AbstractVector; include_additive::Bool=false)
    d = cf.d; r = div(rst.n, d) - 1; p = rst.n_steps
    v = ws.v; c_idx(n) = mod(n, r + 1) + 1
    for i in 0:r
        vi = v[c_idx(-i)]; for q in 1:d; vi[q] = v_in[i*d + q]; end
    end
    for n in 0:p-1
        nn = n + 1
        vnext = zeros(d)
        for (A, k) in cf.det[nn]; mul!(vnext, A, v[c_idx(n-k)], 1.0, 1.0); end
        if include_additive && !isempty(cf.detV); vnext .+= cf.detV[nn]; end
        for q in 1:d; v[c_idx(nn)][q] = vnext[q]; end
    end
    v_out = zeros(length(v_in))
    for i in 0:r
        vv = v[c_idx(p-i)]; for q in 1:d; v_out[i*d + q] = vv[q]; end
    end
    return v_out
end

# ── operators + public API ───────────────────────────────────────────────────
struct MFFactoredOperator{WsT} <: AbstractMatrix{Float64}
    cf::MFFactoredCoefficients; rst::Any; D::Int; ws::WsT
end
Base.size(op::MFFactoredOperator) = (op.D, op.D)
Base.size(op::MFFactoredOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFFactoredOperator) = Float64
LinearAlgebra.issymmetric(op::MFFactoredOperator) = false
LinearAlgebra.ishermitian(op::MFFactoredOperator) = false
function LinearAlgebra.mul!(y::AbstractVector, op::MFFactoredOperator, x::AbstractVector)
    y .= apply_mapping_M2_factored!(op.ws, op.cf, op.rst, x, zeros(op.cf.d*(div(op.rst.n,op.cf.d))),
                                    include_additive=false)
end

struct M1FactoredOperator{WsT} <: AbstractMatrix{Float64}
    cf::MFFactoredCoefficients; rst::Any; D::Int; ws::WsT
end
Base.size(op::M1FactoredOperator) = (op.D, op.D)
Base.size(op::M1FactoredOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::M1FactoredOperator) = Float64
LinearAlgebra.issymmetric(op::M1FactoredOperator) = false
LinearAlgebra.ishermitian(op::M1FactoredOperator) = false
function LinearAlgebra.mul!(y::AbstractVector, op::M1FactoredOperator, x::AbstractVector)
    y .= apply_mapping_M1_factored!(op.ws, op.cf, op.rst, x, include_additive=false)
end

# rst-first API: works directly from the Result, so it NEVER builds the dense
# d²×d² coefficients (the DiscreteMapping_M2_MF constructor's get_all_coefficients
# blows the StaticArrays "expression too large" limit at d ≳ 30). This is the
# path that actually reaches 10-/100-DoF.
function spectralRadiusOfMapping_MF_factored(rst::AbstractResult{d}; args...) where d
    r = div(rst.n, d) - 1
    D = CovVecIdx((r+1)*d).sectionStarts[end]
    cf = get_factored_coefficients(rst; include_additive=false)
    ws = MFFactoredWorkspace(d, r)
    op = MFFactoredOperator(cf, rst, D, ws)
    vals, _, _ = eigsolve(op, rand(D), 1, :LM; args...)
    return abs(vals[1])
end
# convenience: build the Result and solve, bypassing dense coefficients entirely
function spectralRadiusOfMapping_MF_factored(LDDEP::LDDEProblem, method::DiscretizationMethod,
                                             DiscretizationLength::Real; n_steps=nothing, args...)
    rst = n_steps === nothing ?
        calculateResults(LDDEP, method, DiscretizationLength) :
        calculateResults(LDDEP, method, DiscretizationLength; n_steps=n_steps)
    spectralRadiusOfMapping_MF_factored(rst; args...)
end
# delegate the dm form (small d convenience) to the rst path
spectralRadiusOfMapping_MF_factored(dm::stDiscreteMapping_M2_MF; args...) =
    spectralRadiusOfMapping_MF_factored(dm.rst; args...)

function fixPointOfMapping_MF_factored(rst::AbstractResult{d}; args...) where d
    r = div(rst.n, d) - 1
    D1 = (r+1)*d
    D2 = CovVecIdx(D1).sectionStarts[end]
    cf = get_factored_coefficients(rst; include_additive=true)
    ws = MFFactoredWorkspace(d, r)

    op1 = M1FactoredOperator(cf, rst, D1, ws)
    k1 = apply_mapping_M1_factored!(ws, cf, rst, zeros(D1), include_additive=true)
    v_star = gmres(IMinusPhiOperator(op1), k1; reltol=1e-15, args...)

    k2 = apply_mapping_M2_factored!(ws, cf, rst, zeros(D2), v_star, include_additive=true)
    op2 = MFFactoredOperator(cf, rst, D2, ws)
    m_star = gmres(IMinusPhiOperator(op2), k2; reltol=1e-15, args...)
    return m_star
end
fixPointOfMapping_MF_factored(dm::stDiscreteMapping_M2_MF; args...) =
    fixPointOfMapping_MF_factored(dm.rst; args...)
