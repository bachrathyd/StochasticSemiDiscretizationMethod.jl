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

# ── static fast path for small d ─────────────────────────────────────────────
# At small d the factored kernel is dominated not by flops but by dispatch:
# every d×d `mul!` on a Matrix{Float64} costs ~100 ns of BLAS/generic-matmul
# entry overhead versus ~4 ns for an inlined SMatrix product (measured at
# d=2: one apply at p=200 spends 17.7 ms on 232k tiny gemms). The structures
# below mirror the Matrix path 1:1 with SMatrix/SVector entries, so all block
# products inline and the covariance window becomes a flat array of immutable
# blocks. Semantics (circular indexing, staging of cross blocks, the diag
# write LAST so it overwrites the mlag = n-r cross write, m_in/m_out packing)
# are copied exactly from apply_mapping_M2_factored!/_M1_factored!; results
# agree to summation-order rounding only. The Matrix path above remains the
# reference and the d > 8 route (SMatrix compile cost/benefit flips there).

struct MFFactoredCoefficientsS{d,L}
    K::Int
    det::Vector{Vector{Tuple{SMatrix{d,d,Float64,L}, Int}}}
    stoch::Vector{Vector{Tuple{Vector{SMatrix{d,d,Float64,L}}, Int, Int}}}
    detV::Vector{SVector{d,Float64}}
    stochV::Vector{SMatrix{d,d,Float64,L}}
    stochGV::Vector{Vector{Tuple{Vector{SMatrix{d,d,Float64,L}}, Vector{SVector{d,Float64}}, Int, Int}}}
end

function staticize(cf::MFFactoredCoefficients, ::Val{d}) where d
    L = d * d
    S = SMatrix{d,d,Float64,L}
    V = SVector{d,Float64}
    det = [Tuple{S,Int}[(S(A), k) for (A, k) in step] for step in cf.det]
    stoch = [Tuple{Vector{S},Int,Int}[(S.(Gs), k, w) for (Gs, k, w) in step] for step in cf.stoch]
    detV = V[V(x) for x in cf.detV]
    stochV = S[S(M) for M in cf.stochV]
    stochGV = [Tuple{Vector{S},Vector{V},Int,Int}[(S.(Gs), V.(gs), k, w) for (Gs, gs, k, w) in step]
               for step in cf.stochGV]
    return MFFactoredCoefficientsS{d,L}(cf.K, det, stoch, detV, stochV, stochGV)
end

struct MFStaticWorkspace{d,L}
    # covariance window, (r+1)×(r+1) blocks, stored TRANSPOSED: C[b,a] holds the
    # logical block C(a,b), so the hot cross-block sweep (fixed lag row, all m)
    # walks a contiguous column instead of a strided row. Values are identical.
    C::Matrix{SMatrix{d,d,Float64,L}}
    v::Vector{SVector{d,Float64}}
    Cnext::Vector{SMatrix{d,d,Float64,L}}     # staging row for the new cross blocks
end
function MFStaticWorkspace(::Val{d}, r::Int) where d
    L = d * d
    MFStaticWorkspace{d,L}(fill(zero(SMatrix{d,d,Float64,L}), r + 1, r + 1),
                           fill(zero(SVector{d,Float64}), r + 1),
                           fill(zero(SMatrix{d,d,Float64,L}), r + 1))
end

function apply_M2_static!(ws::MFStaticWorkspace{d,L}, cf::MFFactoredCoefficientsS{d,L},
                          rst::AbstractResult, m_in::AbstractVector, v_in::AbstractVector;
                          include_additive::Bool=false) where {d,L}
    K = cf.K
    r = div(rst.n, d) - 1
    p = rst.n_steps
    idx = CovVecIdx((r + 1) * d)
    C = ws.C; v = ws.v; Cnext = ws.Cnext
    c_idx(n) = mod(n, r + 1) + 1
    # The circular index is walked with branch-wrapped cursors below (the hot
    # loops carry no integer div); every cursor value equals the c_idx() of the
    # Matrix path, and all block products accumulate in the same order, so the
    # results are identical to summation rounding.

    # unpack m_in → C, v_in → v  (SMatrix ntuple is column-major: (a,b) ↦ a+(b-1)d);
    # idx is symmetric, so block(j,i) == block(i,j)' bitwise — read i ≤ j only
    ci = 1                                        # c_idx(-i), decrementing
    for i in 0:r
        v[ci] = SVector{d,Float64}(ntuple(q -> v_in[i*d + q], Val(d)))
        cj = ci                                   # c_idx(-j), j from i
        for j in i:r
            B = SMatrix{d,d,Float64,L}(
                ntuple(t -> m_in[idx(i*d + ((t-1) % d + 1), j*d + ((t-1) ÷ d + 1))], Val(L)))
            C[cj, ci] = B                         # transposed storage: [b,a] ↦ (a,b)
            C[ci, cj] = B'
            cj = cj == 1 ? r + 1 : cj - 1
        end
        ci = ci == 1 ? r + 1 : ci - 1
    end

    bn = 1                                        # c_idx(n), advanced per step
    for n in 0:p-1
        nn = n + 1
        det_step = cf.det[nn]
        stoch_step = cf.stoch[nn]
        cn = bn == r + 1 ? 1 : bn + 1             # c_idx(n+1) == c_idx(n-r)

        # v(n+1)
        vnext = zero(SVector{d,Float64})
        for (A, k) in det_step
            row = bn - k; row > 0 || (row += r + 1)   # c_idx(n-k), 0 ≤ k ≤ r
            vnext += A * v[row]
        end
        if include_additive && !isempty(cf.detV)
            vnext += cf.detV[nn]
        end

        # cross blocks C(n+1, m) = Σ_k A_k C(n-k, m)  [+ detV v(m)' if additive]
        for ii in 1:r+1
            Cnext[ii] = zero(SMatrix{d,d,Float64,L})
        end
        for (A, k) in det_step
            row = bn - k; row > 0 || (row += r + 1)
            cm = cn                               # c_idx(mlag), mlag = n-r … n
            for ii in 1:r+1
                Cnext[ii] += A * C[cm, row]       # contiguous column sweep
                cm = cm == r + 1 ? 1 : cm + 1
            end
        end
        if include_additive && !isempty(cf.detV)
            dvn = cf.detV[nn]
            cm = cn
            for ii in 1:r+1
                Cnext[ii] += dvn * v[cm]'
                cm = cm == r + 1 ? 1 : cm + 1
            end
        end

        # diagonal C(n+1,n+1)
        diag = zero(SMatrix{d,d,Float64,L})
        # deterministic  Σ_{k,l} A_k C(n-k,n-l) A_l'
        for (Ak, k) in det_step
            row = bn - k; row > 0 || (row += r + 1)
            for (Al, l) in det_step
                col = bn - l; col > 0 || (col += r + 1)
                diag += Ak * C[col, row] * Al'
            end
        end
        # stochastic (factored)  Σ_{k,l:wk==wl} Σ_m G̃k^m C(n-k,n-l) G̃l^mᵀ
        for (Gk, k, wk) in stoch_step
            row = bn - k; row > 0 || (row += r + 1)
            for (Gl, l, wl) in stoch_step
                wk == wl || continue
                col = bn - l; col > 0 || (col += r + 1)
                Ckl = C[col, row]
                for m in 1:K
                    diag += Gk[m] * Ckl * Gl[m]'
                end
            end
        end
        if include_additive
            if !isempty(cf.detV)
                dv = cf.detV[nn]
                fv = vnext - dv
                diag += dv * fv' + fv * dv' + dv * dv'
            end
            if !isempty(cf.stochV)
                diag += cf.stochV[nn]
            end
            for (Gsamp, gs, k, w) in cf.stochGV[nn]
                row = bn - k; row > 0 || (row += r + 1)
                vk = v[row]
                for m in 1:K
                    # term = (G̃^m v_k) g̃^mᵀ  + transpose
                    Gv = Gsamp[m] * vk
                    diag += Gv * gs[m]' + gs[m] * Gv'
                end
            end
        end

        # commit (diag write LAST: cn == c_idx(n-r) overwrites that cross write)
        v[cn] = vnext
        cm = cn
        for ii in 1:r+1
            C[cm, cn] = Cnext[ii]
            C[cn, cm] = Cnext[ii]'
            cm = cm == r + 1 ? 1 : cm + 1
        end
        C[cn, cn] = diag
        bn = cn
    end

    # extract C → m_out: every half-vectorized slot has a unique (i≤j, a, b)
    # source (i<j: whole block; i==j: a ≤ b), so undef is fully overwritten
    m_out = Vector{Float64}(undef, length(m_in))
    cpi = c_idx(p)                                # c_idx(p-i), decrementing
    for i in 0:r
        cpj = cpi                                 # c_idx(p-j), j from i
        for j in i:r
            Mat = C[cpj, cpi]
            if i == j
                for a in 1:d, b in a:d
                    m_out[idx(i*d + a, j*d + b)] = Mat[a, b]
                end
            else
                for a in 1:d, b in 1:d
                    m_out[idx(i*d + a, j*d + b)] = Mat[a, b]
                end
            end
            cpj = cpj == 1 ? r + 1 : cpj - 1
        end
        cpi = cpi == 1 ? r + 1 : cpi - 1
    end
    return m_out
end

function apply_M1_static!(ws::MFStaticWorkspace{d,L}, cf::MFFactoredCoefficientsS{d,L},
                          rst::AbstractResult, v_in::AbstractVector; include_additive::Bool=false) where {d,L}
    r = div(rst.n, d) - 1; p = rst.n_steps
    v = ws.v; c_idx(n) = mod(n, r + 1) + 1
    for i in 0:r
        v[c_idx(-i)] = SVector{d,Float64}(ntuple(q -> v_in[i*d + q], Val(d)))
    end
    for n in 0:p-1
        nn = n + 1
        vnext = zero(SVector{d,Float64})
        for (A, k) in cf.det[nn]; vnext += A * v[c_idx(n-k)]; end
        if include_additive && !isempty(cf.detV); vnext += cf.detV[nn]; end
        v[c_idx(nn)] = vnext
    end
    v_out = zeros(length(v_in))
    for i in 0:r
        vv = v[c_idx(p-i)]; for q in 1:d; v_out[i*d + q] = vv[q]; end
    end
    return v_out
end

struct MFStaticOperator{d,L} <: AbstractMatrix{Float64}
    cf::MFFactoredCoefficientsS{d,L}; rst::Any; D::Int; ws::MFStaticWorkspace{d,L}
end
Base.size(op::MFStaticOperator) = (op.D, op.D)
Base.size(op::MFStaticOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::MFStaticOperator) = Float64
LinearAlgebra.issymmetric(op::MFStaticOperator) = false
LinearAlgebra.ishermitian(op::MFStaticOperator) = false
function LinearAlgebra.mul!(y::AbstractVector, op::MFStaticOperator{d}, x::AbstractVector) where d
    y .= apply_M2_static!(op.ws, op.cf, op.rst, x, zeros(d*(div(op.rst.n, d))),
                          include_additive=false)
end

struct M1StaticOperator{d,L} <: AbstractMatrix{Float64}
    cf::MFFactoredCoefficientsS{d,L}; rst::Any; D::Int; ws::MFStaticWorkspace{d,L}
end
Base.size(op::M1StaticOperator) = (op.D, op.D)
Base.size(op::M1StaticOperator, i::Int) = i <= 2 ? op.D : 1
Base.eltype(op::M1StaticOperator) = Float64
LinearAlgebra.issymmetric(op::M1StaticOperator) = false
LinearAlgebra.ishermitian(op::M1StaticOperator) = false
function LinearAlgebra.mul!(y::AbstractVector, op::M1StaticOperator, x::AbstractVector)
    y .= apply_M1_static!(op.ws, op.cf, op.rst, x, include_additive=false)
end

# rst-first API: works directly from the Result, so it NEVER builds the dense
# d²×d² coefficients (the DiscreteMapping_M2_MF constructor's get_all_coefficients
# blows the StaticArrays "expression too large" limit at d ≳ 30). This is the
# path that actually reaches 10-/100-DoF.
"""
    spectralRadiusOfMapping_MF_factored(rst::AbstractResult; kwargs...) -> Float64
    spectralRadiusOfMapping_MF_factored(prob::LDDEProblem, method, DiscretizationLength; n_steps, kwargs...) -> Float64

Second-moment spectral radius ``\\rho(\\mathcal{H})`` computed with the
**Kronecker-factored** multiplication-free operator. In addition to the
``\\mathcal{O}(p^2)`` step scaling of [`spectralRadiusOfMapping_MF`](@ref), the
factored It\\^o operator removes the ``\\mathcal{O}(d^4)`` state-dimension wall of
the pre-contracted covariance formulation, making moment-stability analysis of
high-dimensional systems (``d`` up to hundreds of degrees of freedom) tractable.
The result is algebraically identical to the assembled operator; mean-square
stability corresponds to ``\\rho(\\mathcal{H}) < 1``. Keyword arguments are
forwarded to the KrylovKit eigensolver (e.g. `krylovdim`). `return_vec=true`
returns `(ρ, v)` with the converged dominant eigenvector and `x0=v` warm-starts
the eigensolve from it (also makes the otherwise random start deterministic).
"""
function spectralRadiusOfMapping_MF_factored(rst::AbstractResult{d};
                                             x0=nothing, return_vec::Bool=false,
                                             args...) where d
    r = div(rst.n, d) - 1
    D = CovVecIdx((r+1)*d).sectionStarts[end]
    cf = get_factored_coefficients(rst; include_additive=false)
    # warm start: a converged eigenvector from a neighbouring parameter point (map
    # sweeps) typically cuts the matvec count 2-3× vs the default random start.
    v0 = (x0 !== nothing && length(x0) == D) ? Vector{Float64}(x0) : rand(D)
    # eager: stop once the dominant eigenpair meets tol instead of always
    # building the full krylovdim basis (≈ halves the matvec count); a
    # user-passed `eager` in args still wins (later duplicate overrides)
    if d <= 8
        scf = staticize(cf, Val(d))
        sws = MFStaticWorkspace(Val(d), r)
        sop = MFStaticOperator(scf, rst, D, sws)
        vals, vecs, _ = eigsolve(sop, v0, 1, :LM; eager=true, args...)
        return_vec || return abs(vals[1])
        v1 = vecs[1]
        return (abs(vals[1]), eltype(v1)<:Complex ? Float64.(real.(v1)) : Vector{Float64}(v1))
    end
    ws = MFFactoredWorkspace(d, r)
    op = MFFactoredOperator(cf, rst, D, ws)
    vals, vecs, _ = eigsolve(op, v0, 1, :LM; eager=true, args...)
    return_vec || return abs(vals[1])
    v1 = vecs[1]
    (abs(vals[1]), eltype(v1)<:Complex ? Float64.(real.(v1)) : Vector{Float64}(v1))
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

"""
    fixPointOfMapping_MF_factored(rst::AbstractResult; kwargs...) -> Vector{Float64}

Stationary second moment ``\\mathbf{M}^\\ast`` computed with the Kronecker-factored
multiplication-free operator (the fixed-point counterpart of
[`spectralRadiusOfMapping_MF_factored`](@ref)). Returns the stationary covariance
in half-vectorized coordinates; the leading entry is the stationary variance of
the first state component. Requires `calculate_additive = true` when building
`rst`, and mean-square stability (``\\rho(\\mathcal{H}) < 1``).
"""
function fixPointOfMapping_MF_factored(rst::AbstractResult{d}; args...) where d
    r = div(rst.n, d) - 1
    D1 = (r+1)*d
    D2 = CovVecIdx(D1).sectionStarts[end]
    cf = get_factored_coefficients(rst; include_additive=true)
    if d <= 8
        scf = staticize(cf, Val(d))
        sws = MFStaticWorkspace(Val(d), r)
        sop1 = M1StaticOperator(scf, rst, D1, sws)
        sk1 = apply_M1_static!(sws, scf, rst, zeros(D1), include_additive=true)
        sv_star = gmres(IMinusPhiOperator(sop1), sk1; reltol=1e-15, args...)
        sk2 = apply_M2_static!(sws, scf, rst, zeros(D2), sv_star, include_additive=true)
        sop2 = MFStaticOperator(scf, rst, D2, sws)
        return gmres(IMinusPhiOperator(sop2), sk2; reltol=1e-15, args...)
    end
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
