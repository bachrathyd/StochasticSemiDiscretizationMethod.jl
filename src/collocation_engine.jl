# =============================================================================
# collocation_engine.jl — high-order Gauss–Legendre collocation covariance
# engine for linear stochastic delay differential equations.
#
# Ported verbatim (v7 → v8 → v9) from the research code accompanying the paper;
# assembled into a single in-module file. Provides an S-stage Gauss–Legendre
# collocation of the drift with integrated-history states, reaching order 2S in
# the second moment. Depends only on LinearAlgebra and KrylovKit (both `using`d
# by the parent module). Internal API: `Prob`, `build_v9m`, `rho_H_krylov_v9m`,
# `fixPoint_v9m`; user-facing wrappers are in collocation.jl.
# =============================================================================

# ------------------------------------------------------------------ cov_colloc_v7
# =============================================================================
# cov_colloc_v7.jl — v6 + CAUSAL intra-block two-time noise covariance
#
# Provenance: self-contained evolution of demonstration/_highorder_research/
# cov_colloc_v6.jl (kept intact there). v6 verified: noise-off gate exact for
# any B; present-only noise O(h^2S); BUT caps at O(h²) for every GL order the
# moment the delay drift B≠0 (mirror Mathieu benchmark, slope −2 for GL1..GL6).
#
# v7 HYPOTHESIS (from the archived failure analysis): v6 embeds the per-step
# noise increment ΔB with DIAGONAL blocks only (stage-diagonal + endpoint).
# The missing intra-block two-time entries E[η(u_i)η(u_j)ᵀ] are O(h) values;
# r steps later the delayed-drift reads (RowMap ∘ delay stencils) touch those
# entries through quadratic forms with O(h²) weights → O(h²)-per-unit-time
# error → global O(h²) for every S, exactly and only when B≠0 (or via the
# α·K·βᵀ cross once β≠0). The earlier FILL_OFFDIAG test failed because its
# impulse-congruence fill was NON-CAUSAL (b-quadrature over the whole step,
# including noise injected AFTER min(u_i,u_j)).
#
# THE CAUSAL FILL (exact, from Itô): let η(v) be the noise accumulated within
# the current step (η ≡ 0 at all window nodes; delayed reads of η vanish since
# τ spans ≥ 1 step). For u_i < u_j:
#     E[η(u_i) η(u_j)ᵀ] = Δ(u_i,u_i) · Φ_A(u_j, u_i)ᵀ
# where Φ_A is the PRESENT-drift propagator only: ∂v E[η(u_i)η(v)ᵀ] =
# E[η(u_i)η(v)ᵀ]A(v)ᵀ — the delayed-drift term contributes η(v−τ)=0 and
# dW(v) ⊥ η(u_i) for v>u_i. Noise-off ⇒ Δ diag ≡ 0 ⇒ fill ≡ 0 ⇒ the exact
# gate ρ(H)=ρ(U)² is preserved by construction.
#
# Φ_A(u_j,u_i) = Φ(u_j)·Φ(u_i)⁻¹ with Φ(u) the collocation propagator of
# y' = A(t)y from the step start, evaluated at the stage nodes / endpoint.
# =============================================================================

function gl_tab(S)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5] end
    β=[k/sqrt(4k^2-1) for k in 1:S-1]; J=diagm(1=>β)+diagm(-1=>β); E=eigen(Symmetric(J))
    c=(E.values.+1)./2; b=vec(E.vectors[1,:]).^2
    Vm=[c[k]^(j-1) for k in 1:S,j in 1:S]; Aint=[c[i]^j/j for i in 1:S,j in 1:S]
    return Aint/Vm,b,c
end

# Window node catalogue (identical to v6): blocks newest→oldest,
# block k = [endpoint(d); stage1..S(d)], BSIZE=(S+1)d.
function window_nodes(c, r, BSIZE, d)
    S=length(c); times=Float64[]; rs=Int[]
    for k in 0:r
        push!(times, float(k)); push!(rs, k*BSIZE)
        for ss in 1:S; push!(times, (k+1)-c[ss]); push!(rs, k*BSIZE+d+(ss-1)*d); end
    end
    o=sortperm(times); return times[o], rs[o]
end

# High-order delayed read (identical to v6): Lagrange stencil over M nearest
# window nodes; exact node selection when the delayed point aligns with a node.
function delay_read(tb, ntimes, nrows, M, W, d)
    N=length(ntimes)
    idx=searchsortedfirst(ntimes, tb)
    lo=clamp(idx - M÷2, 1, max(1,N-M+1)); hi=min(lo+M-1, N); lo=max(1,hi-M+1)
    sel=lo:hi
    ts=ntimes[sel]
    R=zeros(d,W)
    for (a_i,gi) in enumerate(sel)
        wj=1.0
        for (b_i,gj) in enumerate(sel)
            a_i==b_i && continue
            wj *= (tb-ts[b_i])/(ts[a_i]-ts[b_i])
        end
        base=nrows[gi]
        for di in 1:d; R[di, base+di]+=wj; end
    end
    return R
end

struct Prob
    d::Int; T::Float64; τ::Float64
    A::Function; B::Function; α::Function; β::Function
    σ::Function     # state-independent (additive) noise: t -> d×m matrix, m sources.
                     # Needed only for fixpoint (stationary 2nd-moment) studies: with
                     # purely multiplicative noise (α,β) the homogeneous covariance
                     # recursion C_{n+1}=H(C_n) has no nonzero fixed point — σ adds the
                     # constant forcing D=H(0) that makes the map affine, C*=(I-H)⁻¹D.
end
Prob(d,T,τ,A,B,α,β) = Prob(d,T,τ,A,B,α,β, t->zeros(d,1))

# Per-step data: deterministic collocation block map (identical math to v6's
# step_v4) PLUS the present-drift propagators Φ at the stage nodes & endpoint.
function step_v7(pb::Prob,a,b,c,h,t_n,r)
    d=pb.d; S=length(c); BSIZE=(S+1)*d; W=(r+1)*BSIZE; SD=S*d
    As=[pb.A(t_n+c[i]*h) for i in 1:S]; Bs=[pb.B(t_n+c[i]*h) for i in 1:S]
    αs=[pb.α(t_n+c[i]*h) for i in 1:S]; βs=[pb.β(t_n+c[i]*h) for i in 1:S]
    M=Matrix{Float64}(I,SD,SD)
    for i in 1:S,j in 1:S; M[(i-1)*d+1:i*d,(j-1)*d+1:j*d]-=h*a[i,j]*As[j]; end
    Minv=inv(M)
    Pn=zeros(d,W); for di in 1:d; Pn[di,di]=1.0; end
    ntimes, nrows = window_nodes(c, r, BSIZE, d)
    Mstencil = min(2S+2, length(ntimes))
    Kd=[zeros(d,W) for _ in 1:S]
    for j in 1:S
        s=t_n+c[j]*h; tb=(t_n-(s-pb.τ))/h
        Kd[j]=delay_read(tb, ntimes, nrows, Mstencil, W, d)
    end
    RHS=vcat([Pn for _ in 1:S]...)
    for i in 1:S,j in 1:S; RHS[(i-1)*d+1:i*d,:]+=h*a[i,j]*(Bs[j]*Kd[j]); end
    KY=Minv*RHS
    Ke=copy(Pn); for j in 1:S; Ke+=h*b[j]*(As[j]*KY[(j-1)*d+1:j*d,:]+Bs[j]*Kd[j]); end
    Pblock=vcat(Ke,KY)
    # Present-drift propagators Φ(u) from step start: collocation of Y'=A(t)Y,
    # Y(0)=I → stage values ΦY_k, endpoint Φe. Same stage matrix M.
    Id=Matrix{Float64}(I,d,d)
    RHSΦ=vcat([Id for _ in 1:S]...)
    ΦYstack=Minv*RHSΦ
    ΦY=[ΦYstack[(k-1)*d+1:k*d,:] for k in 1:S]
    Φe=copy(Id); for j in 1:S; Φe+=h*b[j]*(As[j]*ΦY[j]); end
    return (Pblock=Pblock,KY=KY,Kd=Kd,Ke=Ke,Pn=Pn,As=As,Bs=Bs,αs=αs,βs=βs,
            Minv=Minv,ΦY=ΦY,Φe=Φe,
            a=a,b=b,c=c,h=h,d=d,S=S,W=W,BSIZE=BSIZE,r=r)
end

# Per-step noise increment ΔB (BSIZE×BSIZE) for the new block.
# Diagonals exactly as v6: Σ_noise stage solve with Σ0=0, operator = drift +
# α⊗α present self-feedback, source = full de-frozen Egg. Endpoint via the
# b-quadrature. NEW in v7: offdiag=:causal fills every intra-block pair with
# the causal transport Δ(u_i,u_j)=Δ(u_i,u_i)·Φ_A(u_j,u_i)ᵀ; :none reproduces
# v6 (zero off-diagonals) for A/B baseline tests.
function noise_block_v7(st, C; offdiag::Symbol=:causal, cross_on::Bool=true)
    d=st.d; S=st.S; h=st.h; a=st.a; b=st.b; d2=d*d; BSIZE=st.BSIZE
    Id=Matrix{Float64}(I,d,d)
    Egg=Vector{Matrix{Float64}}(undef,S)
    for k in 1:S
        Yk=st.KY[(k-1)*d+1:k*d,:]; Dk=st.Kd[k]
        Mxx=Yk*C*Yk'; Mxd=Yk*C*Dk'; Mdd=Dk*C*Dk'
        α=st.αs[k]; β=st.βs[k]
        cross = cross_on ? (α*Mxd*β' + β*Mxd'*α') : zeros(d,d)
        Egg[k]=α*Mxx*α' + cross + β*Mdd*β'
    end
    Lj=[kron(Id,st.As[j])+kron(st.As[j],Id)+kron(st.αs[j],st.αs[j]) for j in 1:S]
    Mop=Matrix{Float64}(I,S*d2,S*d2)
    for i in 1:S,j in 1:S; Mop[(i-1)*d2+1:i*d2,(j-1)*d2+1:j*d2]-=h*a[i,j]*Lj[j]; end
    rhs=zeros(S*d2)
    for j in 1:S,k in 1:S; rhs[(j-1)*d2+1:j*d2]+=h*a[j,k]*vec(Egg[k]); end
    vΣ=Mop\rhs
    ΔB=zeros(BSIZE,BSIZE)
    endv=zeros(d2)
    for j in 1:S; endv+=h*b[j]*(Lj[j]*vΣ[(j-1)*d2+1:j*d2]+vec(Egg[j])); end
    ΔB[1:d,1:d]=reshape(endv,d,d)
    for k in 1:S
        ΔB[d+(k-1)*d+1:d+k*d, d+(k-1)*d+1:d+k*d]=reshape(vΣ[(k-1)*d2+1:k*d2],d,d)
    end
    if offdiag==:causal
        # node list in time order: stages (c ascending) then endpoint (t=1)
        rows=Vector{UnitRange{Int}}(undef,S+1)
        Δii=Vector{Matrix{Float64}}(undef,S+1)
        Φ  =Vector{Matrix{Float64}}(undef,S+1)
        for k in 1:S
            rows[k]=d+(k-1)*d+1:d+k*d
            Δii[k]=reshape(vΣ[(k-1)*d2+1:k*d2],d,d)
            Φ[k]=st.ΦY[k]
        end
        rows[S+1]=1:d; Δii[S+1]=reshape(endv,d,d); Φ[S+1]=st.Φe
        for i in 1:S+1, j in i+1:S+1        # u_i < u_j in time
            Φrel=Φ[j]/Φ[i]                   # Φ(u_j)·Φ(u_i)⁻¹
            Δij=Δii[i]*Φrel'
            ΔB[rows[i],rows[j]]=Δij
            ΔB[rows[j],rows[i]]=Δij'
        end
    elseif offdiag!=:none
        error("unknown offdiag mode $offdiag")
    end
    return ΔB
end

function build_v7(pb::Prob,S,p)
    a,b,c=gl_tab(S); h=pb.T/p; r=max(round(Int,pb.τ/h),1)
    steps=[step_v7(pb,a,b,c,h,(n-1)*h,r) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    U=Matrix{Float64}(I,W,W)
    for n in 1:p
        Td=zeros(W,W); Td[1:BSIZE,:]=steps[n].Pblock
        for k in 1:r; Td[k*BSIZE+1:(k+1)*BSIZE,(k-1)*BSIZE+1:k*BSIZE]=Matrix(I,BSIZE,BSIZE); end
        U=Td*U
    end
    return (steps=steps,U=U,W=W,BSIZE=BSIZE,p=p)
end

# One period of the second-moment map, structured update (shift + new block):
#   C ↦ [P C Pᵀ + ΔB   (P C)_past ; (C P ᵀ)_past   C_past,past ]
function applyH_v7(eng,C; offdiag::Symbol=:causal, cross_on::Bool=true)
    Ck=copy(C)
    for n in 1:eng.p
        st=eng.steps[n]; W=st.W; BSIZE=st.BSIZE; keep=W-BSIZE
        P=st.Pblock
        PC=P*Ck                                  # BSIZE×W
        newdiag=PC*P' + noise_block_v7(st,Ck; offdiag=offdiag, cross_on=cross_on)
        Cnew=similar(Ck)
        Cnew[1:BSIZE,1:BSIZE]=newdiag
        Cnew[1:BSIZE,BSIZE+1:end]=PC[:,1:keep]
        Cnew[BSIZE+1:end,1:BSIZE]=transpose(PC[:,1:keep])
        Cnew[BSIZE+1:end,BSIZE+1:end]=Ck[1:keep,1:keep]
        Ck=Cnew
    end
    return Ck
end

rho_U_v7(eng)=maximum(abs.(eigen(eng.U).values))

function rho_H_dense(eng; offdiag::Symbol=:causal, cross_on::Bool=true)
    W=eng.W; idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx); H=zeros(Nv,Nv)
    for k in 1:Nv
        (i,j)=idx[k]; C=zeros(W,W); C[i,j]=1.0; C[j,i]=1.0
        Cn=applyH_v7(eng,C; offdiag=offdiag, cross_on=cross_on)
        for mm in 1:Nv; (p2,q2)=idx[mm]; H[mm,k]=Cn[p2,q2]; end
    end
    maximum(abs.(eigen(H).values))
end

function rho_H_krylov(eng; offdiag::Symbol=:causal, cross_on::Bool=true,
                      tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    function vec2sym(v)
        C=zeros(W,W); @inbounds for k in 1:Nv; (i,j)=idx[k]; C[i,j]=v[k]; C[j,i]=v[k]; end; C
    end
    function sym2vec(C)
        v=zeros(Nv); @inbounds for k in 1:Nv; (i,j)=idx[k]; v[k]=C[i,j]; end; v
    end
    op(v)= sym2vec(applyH_v7(eng, vec2sym(v); offdiag=offdiag, cross_on=cross_on))
    x0=sym2vec(Matrix{Float64}(I,W,W))
    vals,_,info = KrylovKit.eigsolve(op, x0, 1, :LM; tol=tol,
                                     krylovdim=min(krylovdim,Nv), maxiter=300)
    return maximum(abs.(vals))
end

# ------------------------------------------------------------------ cov_colloc_v8
# =============================================================================
# cov_colloc_v8.jl — matrix (d≥1) integrated-history engine, time-dependent B.
#
# Extends the validated scalar prototype (cov_colloc_v8_scalar.jl): window
# blocks carry B-WEIGHTED integrated-history DOFs so the delayed drift term
# ∫B(s)x(s−τ)ds is exact even for a Brownian-rough delayed path — the
# mechanism that caps v7 at O(h²) when the delayed drift reads a rough
# component. The weights use the READING step's B(t) (known a priori):
# block over [t_m, t_m+h] stores
#     J_i = ∫_0^{c_i h} B(t_{m+r}+s)·x(t_m+s) ds ,  J_e = ∫_0^h ...
# so step m+r's stage equations use them verbatim:
#     Y_i = x_n + h Σ_j a_ij A_j Y_j + J_i^{(r−1)}
#     x_e = x_n + h Σ_j b_j  A_j Y_j + J_e^{(r−1)}
# (window block k covers [t_{n−k−1}, t_{n−k}] ⇒ the delayed interval is
#  block r−1). Constant B reduces to the scalar prototype's plain integrals.
#
# Noise increment ΔB ((2S+2)d)²: node–node exactly as v7 (Σ_noise stage solve
# with drift+α⊗α, endpoint quadrature, causal fill); node–J and J–J from the
# matrix causal kernel
#     Δ(s,v) = Σn(min)·Ψ(max,min)ᵀ ,  Ψ(a,b) = Φ(a)Φ(b)⁻¹
# integrated with Gauss quadrature on the smooth pieces (split at the kink).
#
# Block layout (BSIZE=(2S+2)d): [x_e; Y_1..Y_S; J_1..J_S; J_e].
# Requires: τ = r·h exactly, single delay, r ≥ 1.
# =============================================================================


# Lagrange basis ℓ_j on nodes c, and its running integral ℓint_j(θ)=∫_0^θ ℓ_j.
# Returns the coefficient matrix once; evaluation helpers below.
function _lagr_coefs(c)
    S=length(c)
    coefs=Vector{Vector{Float64}}(undef,S)     # ascending powers
    for j in 1:S
        coef=[1.0]
        for m in 1:S
            m==j && continue
            newc=zeros(length(coef)+1)
            for (k,ck) in enumerate(coef); newc[k+1]+=ck; newc[k]-=c[m]*ck; end
            coef=newc ./ (c[j]-c[m])
        end
        coefs[j]=coef
    end
    coefs
end
_lint(coef, θ) = sum(ck*θ^k/k for (k,ck) in enumerate(coef))

# 8-point Gauss–Legendre on [0,1]
const _G8 = let
    a,b,c = gl_tab(8); (x=c, w=b)
end

struct StepV8
    Pblock::Matrix{Float64}        # BSIZE×W new-block rows (deterministic)
    Yrows::Matrix{Float64}         # Sd×W stage-value rows (for Egg)
    Dk::Vector{Matrix{Float64}}    # d×W delayed node reads (for Egg)
    As::Vector{Matrix{Float64}}; αs::Vector{Matrix{Float64}}; βs::Vector{Matrix{Float64}}
    σs::Vector{Matrix{Float64}}    # additive-noise loading at the S stage points (d×m each)
    Bf::Function                   # s ∈ [0,h] ↦ B(t_{n+r}+s) (weights of the NEW J's)
    a::Matrix{Float64}; b::Vector{Float64}; c::Vector{Float64}
    lcoef::Vector{Vector{Float64}}
    φstage::Vector{Matrix{Float64}}
    h::Float64; d::Int; S::Int; W::Int; BSIZE::Int; r::Int
end

function step_v8m(pb::Prob, a, b, c, h, t_n, r)
    d=pb.d; S=length(c); BSIZE=(2S+2)*d; W=(r+1)*BSIZE
    As=[Matrix(pb.A(t_n+c[i]*h)) for i in 1:S]
    αs=[Matrix(pb.α(t_n+c[i]*h)) for i in 1:S]
    βs=[Matrix(pb.β(t_n+c[i]*h)) for i in 1:S]
    σs=[Matrix(pb.σ(t_n+c[i]*h)) for i in 1:S]
    Bf = s -> Matrix(pb.B(t_n + r*h + s))       # reader of the block we CREATE
    Id=Matrix{Float64}(I,d,d)
    lcoef=_lagr_coefs(c)
    # window offsets: block k at k*BSIZE; layout [x_e; Y; J_1..J_S; J_e]
    xn_rng      = 1:d                                    # newest endpoint x(t_n)
    delJ(k)     = (r-1)*BSIZE + (S+1)*d + (k-1)*d        # J_k of block r−1 (0-based col offset)
    delJe       = (r-1)*BSIZE + (2S+1)*d
    delY(k)     = (r-1)*BSIZE + d + (k-1)*d              # Y_k of block r−1
    # stage solve: (I − h a⊗A) Ystack = 1⊗x_n + J_del
    M=Matrix{Float64}(I,S*d,S*d)
    for i in 1:S, j in 1:S
        M[(i-1)*d+1:i*d, (j-1)*d+1:j*d] .-= h*a[i,j].*As[j]
    end
    Minv=inv(M)
    RHS=zeros(S*d, W)
    for i in 1:S
        RHS[(i-1)*d+1:i*d, xn_rng] .= Id
        for q in 1:d; RHS[(i-1)*d+q, delJ(i)+q] += 1.0; end
    end
    Yrows=Minv*RHS
    # endpoint row
    erow=zeros(d, W); erow[:, xn_rng] .= Id
    for q in 1:d; erow[q, delJe+q] += 1.0; end
    for j in 1:S
        erow .+= h*b[j].*(As[j]*Yrows[(j-1)*d+1:j*d, :])
    end
    # continuous output K = (a⁻¹ ⊗ I)(Y − 1 x_n)/h
    Ainv=inv(a)
    Krows=zeros(S*d, W)
    for j in 1:S
        for m in 1:S
            Krows[(j-1)*d+1:j*d, :] .+= Ainv[j,m].*Yrows[(m-1)*d+1:m*d, :]
            Krows[(j-1)*d+1:j*d, xn_rng] .-= Ainv[j,m].*Id
        end
    end
    Krows ./= h
    # new J rows: J_i = ∫_0^{θi h} Bf(s)·x(t_n+s) ds with
    # x(θh) = x_n + h Σ_j ℓint_j(θ) K_j  →  Gauss quadrature in s
    θs=vcat(c, 1.0)
    Jrows=zeros((S+1)*d, W)
    for (i,θi) in enumerate(θs)
        Wx = zeros(d,d)                       # weight of x_n
        Wk = [zeros(d,d) for _ in 1:S]        # weight of K_j
        for (gx,gw) in zip(_G8.x, _G8.w)
            s=θi*h*gx; wq=θi*h*gw
            Bs=Bf(s)
            Wx .+= wq.*Bs
            for j in 1:S
                Wk[j] .+= (wq*h*_lint(lcoef[j], s/h)).*Bs
            end
        end
        Jrows[(i-1)*d+1:i*d, xn_rng] .= Wx
        for j in 1:S
            Jrows[(i-1)*d+1:i*d, :] .+= Wk[j]*Krows[(j-1)*d+1:j*d, :]
        end
    end
    Pblock=vcat(erow, Yrows, Jrows)
    # delayed node reads for Egg: x(u_k−τ) = stage node k of block r−1
    Dk=[begin R=zeros(d,W); for q in 1:d; R[q, delY(k)+q]=1.0; end; R end for k in 1:S]
    # drift propagator stage values
    RHSΦ=zeros(S*d, d); for i in 1:S; RHSΦ[(i-1)*d+1:i*d, :] .= Id; end
    Φstack=Minv*RHSΦ
    φstage=[Φstack[(k-1)*d+1:k*d, :] for k in 1:S]
    return StepV8(Pblock, Yrows, Dk, As, αs, βs, σs, Bf, a, b, c, lcoef, φstage,
                  h, d, S, W, BSIZE, r)
end

φ_at_m(st::StepV8, θ) = begin
    Φ = Matrix{Float64}(I, st.d, st.d)
    for j in 1:st.S
        Φ .+= (st.h*_lint(st.lcoef[j], θ)).*(st.As[j]*st.φstage[j])
    end
    Φ
end
function Σn_at_m(st::StepV8, θ, Σs::Vector{Matrix{Float64}}, Egg::Vector{Matrix{Float64}})
    S=st.S; d=st.d
    out=zeros(d,d)
    for j in 1:S
        rhs = st.As[j]*Σs[j] .+ Σs[j]*st.As[j]' .+ st.αs[j]*Σs[j]*st.αs[j]' .+ Egg[j]
        out .+= (st.h*_lint(st.lcoef[j], θ)).*rhs
    end
    out
end
# causal kernel Δ(θs, θv) (units of h): E[η(θs h) η(θv h)ᵀ]
function Δker_m(st::StepV8, θa, θb, Σs, Egg, φcache)
    if θa <= θb
        Σn_at_m(st,θa,Σs,Egg) * (φcache(θb)/φcache(θa))'
    else
        (φcache(θa)/φcache(θb)) * Σn_at_m(st,θb,Σs,Egg)
    end
end

function noise_block_v8m(st::StepV8, C)
    d=st.d; S=st.S; h=st.h; a=st.a; b=st.b; c=st.c; BSIZE=st.BSIZE
    Id=Matrix{Float64}(I,d,d)
    # Egg at stages
    Egg=Vector{Matrix{Float64}}(undef,S)
    for k in 1:S
        Yk=@view st.Yrows[(k-1)*d+1:k*d, :]
        Dk=st.Dk[k]
        Mxx=Yk*C*Yk'; Mxd=Yk*C*Dk'; Mdd=Dk*C*Dk'
        Egg[k]=st.αs[k]*Mxx*st.αs[k]' .+ st.αs[k]*Mxd*st.βs[k]' .+
               st.βs[k]*Mxd'*st.αs[k]' .+ st.βs[k]*Mdd*st.βs[k]' .+
               st.σs[k]*st.σs[k]'                      # state-independent (additive) part
        Egg[k]=(Egg[k].+Egg[k]')./2
    end
    # Σ_noise stage solve: (I − h a⊗L) vecΣ = h a⊗vec(Egg), L=I⊗A+A⊗I+α⊗α
    d2=d*d
    Mop=Matrix{Float64}(I,S*d2,S*d2)
    for i in 1:S, j in 1:S
        Lj=kron(Id,st.As[j]) .+ kron(st.As[j],Id) .+ kron(st.αs[j],st.αs[j])
        Mop[(i-1)*d2+1:i*d2, (j-1)*d2+1:j*d2] .-= h*a[i,j].*Lj
    end
    rhs=zeros(S*d2)
    for j in 1:S, k in 1:S
        rhs[(j-1)*d2+1:j*d2] .+= h*a[j,k].*vec(Egg[k])
    end
    vΣ=Mop\rhs
    Σs=[reshape(vΣ[(k-1)*d2+1:k*d2],d,d) for k in 1:S]
    endm=zeros(d,d)
    for j in 1:S
        endm .+= h*b[j].*(st.As[j]*Σs[j] .+ Σs[j]*st.As[j]' .+ st.αs[j]*Σs[j]*st.αs[j]' .+ Egg[j])
    end
    # φ cache at quadrature/nodes points (memoized on the fly)
    cache=Dict{Float64,Matrix{Float64}}()
    φc(θ) = get!(()->φ_at_m(st,θ), cache, θ)
    # node list: stages c_1..c_S then endpoint (θ=1); ΔB row ranges
    θnode=vcat(c,1.0)
    Δnode=vcat(Σs,[endm])
    rng_node(k)= k<=S ? ((k)*d+1:(k+1)*d) : (1:d)        # Y_k rows or x_e rows
    rng_J(i)   = ((S+1)*d+(i-1)*d+1 : (S+1)*d+i*d)       # J_i rows (i=S+1 → J_e)
    ΔB=zeros(BSIZE,BSIZE)
    # node–node causal fill
    for i in 1:S+1, j in 1:S+1
        θi=θnode[i]; θj=θnode[j]
        V = θi==θj ? Δnode[i] :
            (θi<θj ? Δnode[i]*(φc(θj)/φc(θi))' : (φc(θi)/φc(θj))*Δnode[j])
        ΔB[rng_node(i), rng_node(j)] .= V
    end
    # node–J and J–J: Jη_i = ∫_0^{θa h} Bf(s) η(s) ds
    θJ=vcat(c,1.0)
    # E[Jη_i η(u_k)ᵀ] = ∫ Bf(s) Δ(s,u_k) ds  (split at u_k)
    for i in 1:S+1, k in 1:S+1
        θa=θJ[i]; θk=θnode[k]
        acc=zeros(d,d)
        segs = θk<θa ? ((0.0,θk),(θk,θa)) : ((0.0,θa),)
        for (lo,hi) in segs
            hi<=lo && continue
            for (gx,gw) in zip(_G8.x,_G8.w)
                θ=lo+(hi-lo)*gx
                acc .+= ((hi-lo)*gw).*(st.Bf(θ*h)*Δker_m(st,θ,θk,Σs,Egg,φc))
            end
        end
        V=h.*acc
        ΔB[rng_J(i), rng_node(k)] .= V
        ΔB[rng_node(k), rng_J(i)] .= V'
    end
    # E[Jη_i Jη_jᵀ] = ∬ Bf(s) Δ(s,v) Bf(v)ᵀ  (triangle split in s at v)
    for i in 1:S+1, j in 1:S+1
        j < i && continue
        θa=θJ[i]; θb=θJ[j]
        acc=zeros(d,d)
        for (gx,gw) in zip(_G8.x,_G8.w)
            ϑ=θb*gx; wϑ=θb*gw
            Bv=st.Bf(ϑ*h)
            segs = ϑ<θa ? ((0.0,ϑ),(ϑ,θa)) : ((0.0,θa),)
            for (lo,hi) in segs
                hi<=lo && continue
                for (gx2,gw2) in zip(_G8.x,_G8.w)
                    θ=lo+(hi-lo)*gx2
                    acc .+= (wϑ*(hi-lo)*gw2).*(st.Bf(θ*h)*Δker_m(st,θ,ϑ,Σs,Egg,φc)*Bv')
                end
            end
        end
        V=(h^2).*acc
        ΔB[rng_J(i), rng_J(j)] .= V
        i != j && (ΔB[rng_J(j), rng_J(i)] .= V')
    end
    return ΔB
end

function build_v8m(pb::Prob, S, p)
    a,b,c=gl_tab(S); h=pb.T/p; r=round(Int,pb.τ/h)
    abs(r*h-pb.τ) < 1e-9*max(pb.τ,1.0) || error("τ/h=$(pb.τ/h) not integer")
    r ≥ 1 || error("need r ≥ 1")
    steps=[step_v8m(pb,a,b,c,h,(n-1)*h,r) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    U=Matrix{Float64}(I,W,W)
    for st in steps
        Td=zeros(W,W); Td[1:BSIZE,:]=st.Pblock
        for k in 1:r; Td[k*BSIZE+1:(k+1)*BSIZE,(k-1)*BSIZE+1:k*BSIZE]=Matrix(I,BSIZE,BSIZE); end
        U=Td*U
    end
    return (steps=steps,U=U,W=W,BSIZE=BSIZE,p=p)
end

function applyH_v8m(eng,C)
    Ck=copy(C)
    for st in eng.steps
        W=st.W; BSIZE=st.BSIZE; keep=W-BSIZE
        P=st.Pblock
        PC=P*Ck
        newdiag=PC*P' + noise_block_v8m(st,Ck)
        Cnew=similar(Ck)
        Cnew[1:BSIZE,1:BSIZE]=newdiag
        Cnew[1:BSIZE,BSIZE+1:end]=PC[:,1:keep]
        Cnew[BSIZE+1:end,1:BSIZE]=transpose(PC[:,1:keep])
        Cnew[BSIZE+1:end,BSIZE+1:end]=Ck[1:keep,1:keep]
        Ck=Cnew
    end
    return Ck
end

rho_U_v8m(eng)=maximum(abs.(eigen(eng.U).values))

function rho_H_krylov_v8m(eng; tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    function vec2sym(v)
        C=zeros(W,W); @inbounds for k in 1:Nv; (i,j)=idx[k]; C[i,j]=v[k]; C[j,i]=v[k]; end; C
    end
    function sym2vec(C)
        v=zeros(Nv); @inbounds for k in 1:Nv; (i,j)=idx[k]; v[k]=C[i,j]; end; v
    end
    op(v)= sym2vec(applyH_v8m(eng, vec2sym(v)))
    x0=sym2vec(Matrix{Float64}(I,W,W))
    vals,_,_ = KrylovKit.eigsolve(op, x0, 1, :LM; tol=tol,
                                  krylovdim=min(krylovdim,Nv), maxiter=300)
    return maximum(abs.(vals))
end

# Same as rho_H_krylov_v8m but subtracts the additive-noise constant D=H(0)
# first, so the eigensolve sees the genuinely LINEAR/homogeneous part even
# when pb.σ ≠ 0 (rho_H_krylov_v8m would otherwise feed an AFFINE map into
# KrylovKit.eigsolve, which silently returns a garbage "eigenvalue" polluted
# by the constant drift — use THIS whenever the problem has additive noise).
function rho_Hlin_krylov_v8m(eng; tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    function vec2sym(v)
        C=zeros(W,W); @inbounds for k in 1:Nv; (i,j)=idx[k]; C[i,j]=v[k]; C[j,i]=v[k]; end; C
    end
    function sym2vec(C)
        v=zeros(Nv); @inbounds for k in 1:Nv; (i,j)=idx[k]; v[k]=C[i,j]; end; v
    end
    D = sym2vec(applyH_v8m(eng, zeros(W,W)))
    op(v) = sym2vec(applyH_v8m(eng, vec2sym(v))) .- D
    x0=sym2vec(Matrix{Float64}(I,W,W))
    vals,_,_ = KrylovKit.eigsolve(op, x0, 1, :LM; tol=tol,
                                  krylovdim=min(krylovdim,Nv), maxiter=300)
    return maximum(abs.(vals))
end

# Stationary 2nd-moment fixpoint C* = H(C*): with additive noise (pb.σ ≠ 0)
# applyH_v8m is AFFINE, C_new = Hlin(C) + D with D = applyH_v8m(eng, 0)
# (every state-dependent noise term Mxx/Mxd/Mdd vanishes at C=0, and PC*P' is
# exactly linear), so C* solves (I - Hlin) C* = D — a genuine linear system,
# solved here by GMRES on the vech representation exactly like rho_H_krylov_v8m.
function fixPoint_v8m(eng; tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    function vec2sym(v)
        C=zeros(W,W); @inbounds for k in 1:Nv; (i,j)=idx[k]; C[i,j]=v[k]; C[j,i]=v[k]; end; C
    end
    function sym2vec(C)
        v=zeros(Nv); @inbounds for k in 1:Nv; (i,j)=idx[k]; v[k]=C[i,j]; end; v
    end
    D = applyH_v8m(eng, zeros(W,W))
    dvec = sym2vec(D)
    Hlin(v) = sym2vec(applyH_v8m(eng, vec2sym(v))) .- dvec
    sol, info = KrylovKit.linsolve(v -> v .- Hlin(v), dvec, dvec; tol=tol,
                                    krylovdim=min(krylovdim,Nv), maxiter=300)
    info.converged == 0 && @warn "fixPoint_v8m: GMRES did not fully converge" info
    return vec2sym(sol)
end
# scalar summary for cross-method comparison: stationary variance of the
# newest state's first component (both engines put x(t_current) in rows 1:d)
fixPointVar1_v8m(eng; args...) = fixPoint_v8m(eng; args...)[1,1]


# ------------------------------------------------------------------ cov_colloc_v9
# =============================================================================
# cov_colloc_v9.jl — DOF-pruned integrated-history engine.
#
# v8 persists (2S+2) sub-blocks per delay slot: [x_e; Y_1..Y_S; J_1..J_S; J_e].
# The covariance therefore costs (2S+2)^2 relative to classical SDM (1 block).
# But the S stage-value blocks Y_i are read ONLY by the delayed MULTIPLICATIVE
# noise term (β·x(t−τ) sampled at the stage nodes). When the problem has no
# delayed multiplicative noise (β ≡ 0 over the whole period — the common case
# for delayed feedback control, turning/milling, and any problem whose noise
# reads only the present state), the persistent Y_i are never read and can be
# dropped. The block shrinks to [x_e; J_1..J_S; J_e] of size (S+2)d, cutting the
# covariance factor from (2S+2)^2 to (S+2)^2 (e.g. S=4: 100 → 36, ≈2.8×).
#
# The stage values are STILL computed inside each step (they build the J
# integrals, the endpoint, and the present-noise contraction) — they are simply
# not carried in the persistent covariance window. All order-critical machinery
# (integrated history J, causal intra-block noise fill) is unchanged, so v9
# reproduces v8 to solver tolerance whenever β ≡ 0.
#
# When β ≢ 0 the reduction is unsafe (the delayed noise genuinely reads the
# stage values); build_v9m then transparently falls back to build_v8m.
#
# Block layout (BSIZE9 = (S+2)d): [x_e; J_1..J_S; J_e]. Requires τ = r·h, r ≥ 1.
# =============================================================================

struct StepV9
    Pblock::Matrix{Float64}        # BSIZE9×W (deterministic, reduced block)
    Yrows::Matrix{Float64}         # Sd×W stage-value rows (computed, NOT persisted)
    As::Vector{Matrix{Float64}}; αs::Vector{Matrix{Float64}}
    σs::Vector{Matrix{Float64}}
    Bf::Function
    a::Matrix{Float64}; b::Vector{Float64}; c::Vector{Float64}
    lcoef::Vector{Vector{Float64}}
    φstage::Vector{Matrix{Float64}}
    h::Float64; d::Int; S::Int; W::Int; BSIZE::Int; r::Int
end

# β ≡ 0 test over a fine sample of the period
function _no_delay_noise(pb::Prob; nt=64)
    for k in 0:nt-1
        maximum(abs, pb.β((k+0.5)/nt * pb.T)) > 1e-14 && return false
    end
    true
end

function step_v9m(pb::Prob, a, b, c, h, t_n, r)
    d=pb.d; S=length(c); BSIZE=(S+2)*d; W=(r+1)*BSIZE
    As=[Matrix(pb.A(t_n+c[i]*h)) for i in 1:S]
    αs=[Matrix(pb.α(t_n+c[i]*h)) for i in 1:S]
    σs=[Matrix(pb.σ(t_n+c[i]*h)) for i in 1:S]
    Bf = s -> Matrix(pb.B(t_n + r*h + s))
    Id=Matrix{Float64}(I,d,d)
    lcoef=_lagr_coefs(c)
    xn_rng = 1:d
    # reduced-block delayed offsets (J's start at offset d, no Y block)
    delJ(k) = (r-1)*BSIZE + (k)*d              # J_k of block r−1 (k=1..S): offset (k)*d
    delJe   = (r-1)*BSIZE + (S+1)*d            # J_e of block r−1
    M=Matrix{Float64}(I,S*d,S*d)
    for i in 1:S, j in 1:S; M[(i-1)*d+1:i*d,(j-1)*d+1:j*d] .-= h*a[i,j].*As[j]; end
    Minv=inv(M)
    RHS=zeros(S*d, W)
    for i in 1:S
        RHS[(i-1)*d+1:i*d, xn_rng] .= Id
        for q in 1:d; RHS[(i-1)*d+q, delJ(i)+q] += 1.0; end
    end
    Yrows=Minv*RHS
    erow=zeros(d, W); erow[:, xn_rng] .= Id
    for q in 1:d; erow[q, delJe+q] += 1.0; end
    for j in 1:S; erow .+= h*b[j].*(As[j]*Yrows[(j-1)*d+1:j*d, :]); end
    Ainv=inv(a)
    Krows=zeros(S*d, W)
    for j in 1:S, m in 1:S
        Krows[(j-1)*d+1:j*d, :] .+= Ainv[j,m].*Yrows[(m-1)*d+1:m*d, :]
        Krows[(j-1)*d+1:j*d, xn_rng] .-= Ainv[j,m].*Id
    end
    Krows ./= h
    θs=vcat(c, 1.0)
    Jrows=zeros((S+1)*d, W)
    for (i,θi) in enumerate(θs)
        Wx = zeros(d,d); Wk = [zeros(d,d) for _ in 1:S]
        for (gx,gw) in zip(_G8.x, _G8.w)
            s=θi*h*gx; wq=θi*h*gw; Bs=Bf(s)
            Wx .+= wq.*Bs
            for j in 1:S; Wk[j] .+= (wq*h*_lint(lcoef[j], s/h)).*Bs; end
        end
        Jrows[(i-1)*d+1:i*d, xn_rng] .= Wx
        for j in 1:S; Jrows[(i-1)*d+1:i*d, :] .+= Wk[j]*Krows[(j-1)*d+1:j*d, :]; end
    end
    Pblock=vcat(erow, Jrows)                    # [x_e; J_1..J_S; J_e], no Y
    RHSΦ=zeros(S*d, d); for i in 1:S; RHSΦ[(i-1)*d+1:i*d, :] .= Id; end
    Φstack=Minv*RHSΦ
    φstage=[Φstack[(k-1)*d+1:k*d, :] for k in 1:S]
    return StepV9(Pblock, Yrows, As, αs, σs, Bf, a, b, c, lcoef, φstage,
                  h, d, S, W, BSIZE, r)
end

# helpers mirroring the v8 versions but taking a StepV9
_φ_at9(st::StepV9, θ) = begin
    Φ=Matrix{Float64}(I, st.d, st.d)
    for j in 1:st.S; Φ .+= (st.h*_lint(st.lcoef[j], θ)).*(st.As[j]*st.φstage[j]); end
    Φ
end
_Σn_at9(st::StepV9, θ, Σs, Egg) = begin
    out=zeros(st.d,st.d)
    for j in 1:st.S
        rhs = st.As[j]*Σs[j] .+ Σs[j]*st.As[j]' .+ st.αs[j]*Σs[j]*st.αs[j]' .+ Egg[j]
        out .+= (st.h*_lint(st.lcoef[j], θ)).*rhs
    end
    out
end
_Δker9(st::StepV9, θa, θb, Σs, Egg, φc) =
    θa<=θb ? _Σn_at9(st,θa,Σs,Egg)*(φc(θb)/φc(θa))' :
             (φc(θa)/φc(θb))*_Σn_at9(st,θb,Σs,Egg)

function noise_block_v9m(st::StepV9, C)
    d=st.d; S=st.S; h=st.h; a=st.a; b=st.b; c=st.c; BSIZE=st.BSIZE
    Id=Matrix{Float64}(I,d,d)
    # present-state contraction only (β ≡ 0 ⇒ no delayed-node reads)
    Egg=Vector{Matrix{Float64}}(undef,S)
    for k in 1:S
        Yk=@view st.Yrows[(k-1)*d+1:k*d, :]
        Mxx=Yk*C*Yk'
        e = st.αs[k]*Mxx*st.αs[k]' .+ st.σs[k]*st.σs[k]'
        Egg[k]=(e.+e')./2
    end
    d2=d*d
    Mop=Matrix{Float64}(I,S*d2,S*d2)
    for i in 1:S, j in 1:S
        Lj=kron(Id,st.As[j]) .+ kron(st.As[j],Id) .+ kron(st.αs[j],st.αs[j])
        Mop[(i-1)*d2+1:i*d2, (j-1)*d2+1:j*d2] .-= h*a[i,j].*Lj
    end
    rhs=zeros(S*d2)
    for j in 1:S, k in 1:S; rhs[(j-1)*d2+1:j*d2] .+= h*a[j,k].*vec(Egg[k]); end
    vΣ=Mop\rhs
    Σs=[reshape(vΣ[(k-1)*d2+1:k*d2],d,d) for k in 1:S]
    endm=zeros(d,d)
    for j in 1:S
        endm .+= h*b[j].*(st.As[j]*Σs[j] .+ Σs[j]*st.As[j]' .+ st.αs[j]*Σs[j]*st.αs[j]' .+ Egg[j])
    end
    cache=Dict{Float64,Matrix{Float64}}(); φc(θ)=get!(()->_φ_at9(st,θ),cache,θ)
    # reduced block DOFs: x_e (rows 1:d) then J_1..J_S, J_e
    rng_J(i) = (i*d+1 : (i+1)*d)               # i=1..S+1 (i=S+1 → J_e)
    ΔB=zeros(BSIZE,BSIZE)
    ΔB[1:d, 1:d] .= endm                        # x_e – x_e (endpoint noise)
    θJ=vcat(c,1.0)
    # E[Jη_i η(x_e)]  — x_e is the endpoint node θ=1 (θk=1 ≥ θa ⇒ single segment)
    for i in 1:S+1
        θa=θJ[i]; acc=zeros(d,d)
        for (gx,gw) in zip(_G8.x,_G8.w)
            θ=θa*gx
            acc .+= (θa*gw).*(st.Bf(θ*h)*_Δker9(st,θ,1.0,Σs,Egg,φc))
        end
        V=h.*acc
        ΔB[rng_J(i), 1:d] .= V; ΔB[1:d, rng_J(i)] .= V'
    end
    # E[Jη_i Jη_j]
    for i in 1:S+1, j in 1:S+1
        j < i && continue
        θa=θJ[i]; θb=θJ[j]; acc=zeros(d,d)
        for (gx,gw) in zip(_G8.x,_G8.w)
            ϑ=θb*gx; wϑ=θb*gw; Bv=st.Bf(ϑ*h)
            segs = ϑ<θa ? ((0.0,ϑ),(ϑ,θa)) : ((0.0,θa),)
            for (lo,hi) in segs
                hi<=lo && continue
                for (gx2,gw2) in zip(_G8.x,_G8.w)
                    θ=lo+(hi-lo)*gx2
                    acc .+= (wϑ*(hi-lo)*gw2).*(st.Bf(θ*h)*_Δker9(st,θ,ϑ,Σs,Egg,φc)*Bv')
                end
            end
        end
        V=(h^2).*acc
        ΔB[rng_J(i), rng_J(j)] .= V
        i != j && (ΔB[rng_J(j), rng_J(i)] .= V')
    end
    return ΔB
end

# ---------------------------------------------------------------------------
# Precomputed per-step noise operator (fast path for the Krylov eigensolve).
#
# noise_block_v9m(st, C) is AFFINE in the covariance C: C enters ONLY through the
# S stage contractions Egg[k] = αs[k]·(Yk C Ykᵀ)·αs[k]ᵀ (+ σσᵀ, constant).
# Everything downstream — the Σ-noise stage solve, the present-drift propagators
# φ, and the Bf-weighted causal-kernel quadrature — is a FIXED linear map that
# noise_block_v9m rebuilds on EVERY Krylov matvec (≈1000 Bf evaluations + a
# Float64-keyed φ Dict + megabytes of allocation per call; ~99% of the solver
# time). We precompute that map once per step.
#
# Downstream of the Σ-solve every block of ΔB is linear in the S matrices
#     R_m = As_m Σs_m + Σs_m As_mᵀ + αs_m Σs_m αs_mᵀ + Egg_m ,   Σs = Mop⁻¹ rhs(Egg),
# so for each output block (br,bc) we store S matrices `Mten[g][m]` (d²×d²) with
#     vec(ΔB[br,bc]) = Σ_m Mten[g][m] · vec(R_m) .
# Per matvec the noise block is then: Egg (tiny), one S·d²×S·d² solve, and a
# handful of d²×d² mat-vecs — numerically identical to noise_block_v9m
# (agreement ~1e-15), ~1000× cheaper.
struct NoiseOpV9
    Mop::Matrix{Float64}                       # S·d² Σ-solve operator (C-independent)
    a::Matrix{Float64}
    As::Vector{Matrix{Float64}}
    αs::Vector{Matrix{Float64}}
    σs::Vector{Matrix{Float64}}
    # The stage rows Y read the covariance window only at their structurally
    # nonzero columns (the newest endpoint x_e and the S delayed-J blocks —
    # (S+1)d columns out of W), so the Egg contraction Y C Yᵀ needs just the
    # gathered nzc×nzc submatrix of C instead of two full-W products.
    nzc::Vector{Int}                           # nonzero columns of Yrows (logical)
    Ynz::Matrix{Float64}                       # Sd × length(nzc) compressed stage rows
    h::Float64; d::Int; S::Int; BSIZE::Int
    brs::Vector{UnitRange{Int}}                # output block row/col ranges …
    bcs::Vector{UnitRange{Int}}
    Mten::Vector{Vector{Matrix{Float64}}}      # … and their response tensors
end

# M .+= s·kron(transpose(R), L)  (all d×d), allocation-free.
@inline function _kron_acc!(M, s, R, L)
    d = size(L, 1)
    @inbounds for jc in 1:d, ic in 1:d
        v = s * R[jc, ic]                       # transpose(R)[ic,jc] = R[jc,ic]
        r0 = (ic-1)*d; c0 = (jc-1)*d
        for bb in 1:d, aa in 1:d
            M[r0+aa, c0+bb] += v * L[aa, bb]
        end
    end
end

# Same but takes RT = Rᵀ (lets the callers build RT with a single mul! instead
# of materializing transpose products): M .+= s·kron(RT, L).
@inline function _kron_acc_t!(M, s, RT, L)
    d = size(L, 1)
    @inbounds for jc in 1:d, ic in 1:d
        v = s * RT[ic, jc]
        r0 = (ic-1)*d; c0 = (jc-1)*d
        for bb in 1:d, aa in 1:d
            M[r0+aa, c0+bb] += v * L[aa, bb]
        end
    end
end

function _build_noiseop_v9(st::StepV9)
    d=st.d; S=st.S; h=st.h; a=st.a; b=st.b; c=st.c; BSIZE=st.BSIZE; d2=d*d
    Id=Matrix{Float64}(I,d,d)
    # C-independent Σ-noise operator (I − h a⊗L), L=I⊗A+A⊗I+α⊗α
    Mop=Matrix{Float64}(I,S*d2,S*d2)
    for i in 1:S, j in 1:S
        Lj=kron(Id,st.As[j]) .+ kron(st.As[j],Id) .+ kron(st.αs[j],st.αs[j])
        Mop[(i-1)*d2+1:i*d2, (j-1)*d2+1:j*d2] .-= h*a[i,j].*Lj
    end
    # memoized present-drift propagator and its inverse (both C-independent).
    # The causal kernel needs ratios φ(a)/φ(b); precompute φ⁻¹ once per node and
    # multiply, avoiding ~1000 small dense right-divisions (LU solves) per step.
    φcache=Dict{Float64,Matrix{Float64}}();  φf(θ)=get!(()->_φ_at9(st,θ),φcache,θ)
    iφcache=Dict{Float64,Matrix{Float64}}(); iφf(θ)=get!(()->inv(φf(θ)),iφcache,θ)
    Bcache=Dict{Float64,Matrix{Float64}}();  Bfc(θ)=get!(()->st.Bf(θ*h),Bcache,θ)
    T1=Matrix{Float64}(undef,d,d); T2=Matrix{Float64}(undef,d,d)   # gemm scratch
    brs=UnitRange{Int}[]; bcs=UnitRange{Int}[]; Mten=Vector{Matrix{Float64}}[]
    function newgroup!(br,bc)
        push!(brs,br); push!(bcs,bc); push!(Mten,[zeros(d2,d2) for _ in 1:S]); length(brs)
    end
    coefΣ(θ,m) = h*_lint(st.lcoef[m], θ)        # Σn(θ) = Σ_m coefΣ(θ,m)·R_m
    rng_J(i) = (i*d+1 : (i+1)*d)
    # endpoint–endpoint block:  endm = Σ_m (h b_m) R_m
    g=newgroup!(1:d, 1:d)
    for m in 1:S; _kron_acc!(Mten[g][m], h*b[m], Id, Id); end
    θJ=vcat(c, 1.0)
    # node–J blocks: ΔB[J_i, x_e] = ∫ Bf(s) Σn(s) (φ(1)/φ(s))ᵀ ds  (x_e is node θ=1)
    # (all products staged through the T1/T2 scratch as Rᵀ; see _kron_acc_t!)
    for i in 1:S+1
        θa=θJ[i]; g=newgroup!(rng_J(i), 1:d)
        for (gx,gw) in zip(_G8.x, _G8.w)
            θ=θa*gx
            L=Bfc(θ); mul!(T1, φf(1.0), iφf(θ))              # RT = φ(1)·φ(θ)⁻¹
            for m in 1:S; _kron_acc_t!(Mten[g][m], h*θa*gw*coefΣ(θ,m), T1, L); end
        end
    end
    # J–J blocks: ΔB[J_i, J_j] = ∬ Bf(s) Δ(s,v) Bf(v)ᵀ  (causal kernel, split at v)
    for i in 1:S+1, j in 1:S+1
        j < i && continue
        θa=θJ[i]; θb=θJ[j]; g=newgroup!(rng_J(i), rng_J(j))
        for (gx,gw) in zip(_G8.x, _G8.w)
            ϑ=θb*gx; wϑ=θb*gw; Bv=Bfc(ϑ)
            segs = ϑ<θa ? ((0.0,ϑ),(ϑ,θa)) : ((0.0,θa),)
            for (lo,hi) in segs
                hi<=lo && continue
                for (gx2,gw2) in zip(_G8.x, _G8.w)
                    θ=lo+(hi-lo)*gx2
                    w=h*h*wϑ*(hi-lo)*gw2
                    if θ<=ϑ                       # Δ = Σn(θ)·(φ(ϑ)/φ(θ))ᵀ
                        L=Bfc(θ)
                        mul!(T1, φf(ϑ), iφf(θ)); mul!(T2, Bv, T1)   # RT = Bv·φϑ·φθ⁻¹
                        for m in 1:S; _kron_acc_t!(Mten[g][m], w*coefΣ(θ,m), T2, L); end
                    else                          # Δ = (φ(θ)/φ(ϑ))·Σn(ϑ)
                        mul!(T1, φf(θ), iφf(ϑ)); mul!(T2, Bfc(θ), T1)  # L = Bfθ·φθ·φϑ⁻¹
                        for m in 1:S; _kron_acc_t!(Mten[g][m], w*coefΣ(ϑ,m), Bv, T2); end
                    end
                end
            end
        end
    end
    nzc = [j for j in 1:size(st.Yrows,2) if any(!iszero, @view st.Yrows[:,j])]
    Ynz = st.Yrows[:, nzc]
    NoiseOpV9(Mop, a, st.As, st.αs, st.σs, nzc, Ynz, h, d, S, BSIZE, brs, bcs, Mten)
end

# Assemble ΔB for a given covariance C using the precomputed operator.
# `_noise_apply_v9!` overwrites its target; `_noise_apply_add_v9!` accumulates
# (used to fold the noise block straight into the new-block diagonal).
function _noise_apply_v9!(ΔB, op::NoiseOpV9, C)
    fill!(ΔB, 0.0)
    _noise_apply_add_v9!(ΔB, op, C)
end

function _noise_apply_add_v9!(target, op::NoiseOpV9, C)
    nnz=length(op.nzc)
    Cnz=Matrix{Float64}(undef, nnz, nnz)
    @inbounds for (jj,cj) in enumerate(op.nzc), (ii,ci) in enumerate(op.nzc)
        Cnz[ii,jj]=C[ci,cj]
    end
    _noise_apply_add_nz_v9!(target, op, Cnz)
end

# Same, but C is in ring-buffer (rotated) layout: `phys[t]` is the physical
# column of the t-th noise-gather column op.nzc[t]; `Cnz` is a caller-owned
# nnz×nnz scratch (avoids the per-call allocation on the solver hot path).
function _noise_apply_add_v9_phys!(target, op::NoiseOpV9, C, phys::AbstractVector{Int}, Cnz)
    @inbounds for jj in eachindex(phys), ii in eachindex(phys)
        Cnz[ii,jj]=C[phys[ii],phys[jj]]
    end
    _noise_apply_add_nz_v9!(target, op, Cnz)
end

function _noise_apply_add_nz_v9!(target, op::NoiseOpV9, Cnz)
    d=op.d; S=op.S; d2=d*d
    Egg=Vector{Matrix{Float64}}(undef,S)
    for k in 1:S
        Yk=@view op.Ynz[(k-1)*d+1:k*d, :]
        Mxx=Yk*Cnz*Yk'
        e=op.αs[k]*Mxx*op.αs[k]' .+ op.σs[k]*op.σs[k]'
        Egg[k]=(e.+e')./2
    end
    rhs=zeros(S*d2)
    for j in 1:S, k in 1:S; rhs[(j-1)*d2+1:j*d2] .+= op.h*op.a[j,k].*vec(Egg[k]); end
    vΣ=op.Mop\rhs
    Rm=[Vector{Float64}(undef,d2) for _ in 1:S]
    for m in 1:S
        Σm=reshape(@view(vΣ[(m-1)*d2+1:m*d2]), d, d)
        R=op.As[m]*Σm .+ Σm*op.As[m]' .+ op.αs[m]*Σm*op.αs[m]' .+ Egg[m]
        Rm[m].=vec(R)
    end
    vb=Vector{Float64}(undef,d2)
    @inbounds for g in eachindex(op.brs)
        fill!(vb, 0.0)
        for m in 1:S; mul!(vb, op.Mten[g][m], Rm[m], 1.0, 1.0); end
        br=op.brs[g]; bc=op.bcs[g]
        k=0
        for cc in bc, rr in br
            k+=1; target[rr,cc]+=vb[k]
        end
        if br!=bc
            k=0
            for cc in bc, rr in br
                k+=1; target[cc,rr]+=vb[k]     # transpose block
            end
        end
    end
    target
end

function build_v9m(pb::Prob, S, p; force=false)
    if !_no_delay_noise(pb)
        force || return build_v8m(pb, S, p)      # unsafe to prune ⇒ fall back to v8
        @warn "build_v9m(force=true) on a problem with β ≢ 0: the pruned engine " *
              "drops the stage-value blocks, so the delayed multiplicative noise " *
              "is IGNORED — diagnostics only, ρ will be wrong" maxlog=1
    end
    a,b,c=gl_tab(S); h=pb.T/p; r=round(Int,pb.τ/h)
    abs(r*h-pb.τ) < 1e-9*max(pb.τ,1.0) || error("τ/h not integer")
    r ≥ 1 || error("need r ≥ 1")
    steps=[step_v9m(pb,a,b,c,h,(n-1)*h,r) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    # precompute the per-step noise operators (the eigensolve's hot path)
    ops=[_build_noiseop_v9(st) for st in steps]
    # NOTE: the dense W×W monodromy `U` (rho_U) is intentionally NOT assembled
    # here — the second-moment path (rho_H_krylov_v9m / fixPoint_v9m) never reads
    # it, and building it was O(p·W³) of wasted work.
    return (steps=steps,ops=ops,W=W,BSIZE=BSIZE,p=p,r=r,d=steps[1].d,engine=:v9)
end

function applyH_v9m(eng,C)
    Ck=copy(C)
    haskey(eng, :ops) || return _applyH_v9m_slow(eng, Ck)   # legacy engines w/o ops
    ΔB=Matrix{Float64}(undef, eng.BSIZE, eng.BSIZE)
    for n in 1:eng.p
        st=eng.steps[n]; W=st.W; BSIZE=st.BSIZE; keep=W-BSIZE
        P=st.Pblock; PC=P*Ck
        _noise_apply_v9!(ΔB, eng.ops[n], Ck)
        newdiag=PC*P' + ΔB
        Cnew=similar(Ck)
        Cnew[1:BSIZE,1:BSIZE]=newdiag
        Cnew[1:BSIZE,BSIZE+1:end]=PC[:,1:keep]
        Cnew[BSIZE+1:end,1:BSIZE]=transpose(PC[:,1:keep])
        Cnew[BSIZE+1:end,BSIZE+1:end]=Ck[1:keep,1:keep]
        Ck=Cnew
    end
    return Ck
end

# Original (reference) path, kept for regression comparison against the
# precomputed noise operator; rebuilds noise_block_v9m every call.
function _applyH_v9m_slow(eng,C)
    Ck=copy(C)
    for st in eng.steps
        W=st.W; BSIZE=st.BSIZE; keep=W-BSIZE
        P=st.Pblock; PC=P*Ck
        newdiag=PC*P' + noise_block_v9m(st,Ck)
        Cnew=similar(Ck)
        Cnew[1:BSIZE,1:BSIZE]=newdiag
        Cnew[1:BSIZE,BSIZE+1:end]=PC[:,1:keep]
        Cnew[BSIZE+1:end,1:BSIZE]=transpose(PC[:,1:keep])
        Cnew[BSIZE+1:end,BSIZE+1:end]=Ck[1:keep,1:keep]
        Ck=Cnew
    end
    return Ck
end

# dispatch helpers: v9/vT engines carry :engine; v8 fallback is a plain NamedTuple
_applyH(eng,C) = !haskey(eng,:engine) ? applyH_v8m(eng,C) :
                 eng.engine==:v9      ? applyH_v9m(eng,C) :
                 eng.engine==:vT      ? applyH_vT(eng,C)  : applyH_v8m(eng,C)

# ---------------------------------------------------------------------------
# Allocation-free one-period map for the Krylov eigensolve.
#
# The structured update  C ↦ [PCPᵀ+ΔB  (PC)_past; (CPᵀ)_past  C_past,past]  is
# applied on a BLOCK RING BUFFER: the covariance stays in place and a rotation
# offset tracks which physical (r+1)-block slot holds which window age. Each
# step overwrites only the slot whose block just fell out of the window (one
# B×B diagonal block + 2·B·(W−B) cross entries); the O(W²) "shift the whole
# history" copy of a naive implementation never happens. `Pblock`'s column
# sparsity (the new-block rows read the window only at the newest endpoint,
# cols 1:d, and the delayed J-block r−1 — two contiguous ranges) makes P·C
# cost O(B²·W) instead of O(B·W²).
struct V9Workspace
    C::Matrix{Float64}        # W×W covariance, block ring-buffer layout
    PC::Matrix{Float64}       # B×W new-block cross rows (physical cols)
    nd::Matrix{Float64}       # B×B new-diagonal scratch
    physnz::Vector{Int}       # per-step physical noise-gather columns
    Cnz::Matrix{Float64}      # nnz×nnz gathered submatrix for the noise op
    lmap::Vector{Int}         # logical→physical column map (period-end offset)
end
V9Workspace(eng) = begin
    nnz = maximum(length(o.nzc) for o in eng.ops)   # per-step nzc counts can differ
    V9Workspace(zeros(eng.W, eng.W), zeros(eng.BSIZE, eng.W),
                zeros(eng.BSIZE, eng.BSIZE), Vector{Int}(undef, nnz),
                Matrix{Float64}(undef, nnz, nnz), Vector{Int}(undef, eng.W))
end

# logical column c ↦ physical column at rotation offset o (0-based blocks)
function _fill_lmap!(lmap, o, B, nblk)
    @inbounds for c in eachindex(lmap)
        b = (c-1) ÷ B; w = c - b*B
        lmap[c] = ((o + b) % nblk)*B + w
    end
    lmap
end

# One period on the ring buffer. ws.C enters in CANONICAL layout (offset 0)
# and leaves rotated; returns the final offset (always mod(-p, r+1)).
function _applyH_period_ring!(ws::V9Workspace, eng)
    B=eng.BSIZE; d=eng.d; r=eng.r; nblk=r+1
    C=ws.C; PC=ws.PC; nd=ws.nd; phys=ws.physnz
    delcols = (r-1)*B+d+1 : r*B                   # delayed J-cols of Pblock
    o = 0
    @inbounds for n in 1:eng.p
        P=eng.steps[n].Pblock; op=eng.ops[n]
        xe_b  = o                                 # physical block of logical 0
        del_b = (o + r - 1) % nblk                # physical block of logical r−1
        new_b = (o + r) % nblk                    # dropped slot ⇒ new block
        xe_rows  = xe_b*B+1 : xe_b*B+d
        del_rows = del_b*B+d+1 : del_b*B+B
        # PC = P[:,1:d]·C[x_e rows,:] + P[:,delcols]·C[del rows,:]
        @views mul!(PC, P[:,1:d],     C[xe_rows,:],  1.0, 0.0)
        @views mul!(PC, P[:,delcols], C[del_rows,:], 1.0, 1.0)
        # new diagonal block = PC·Pᵀ (same two column ranges) + noise
        @views mul!(nd, PC[:,xe_rows],  transpose(P[:,1:d]),     1.0, 0.0)
        @views mul!(nd, PC[:,del_rows], transpose(P[:,delcols]), 1.0, 1.0)
        nnz=length(op.nzc)
        for t in 1:nnz
            c=op.nzc[t]; b=(c-1)÷B; w=c-b*B
            phys[t] = ((o + b) % nblk)*B + w
        end
        @views _noise_apply_add_v9_phys!(nd, op, C, phys[1:nnz], ws.Cnz[1:nnz,1:nnz])
        # overwrite the dead slot with the new block (reads of C are done)
        nrng = new_b*B+1 : new_b*B+B
        for q in 0:nblk-1
            q == new_b && continue
            qrng = q*B+1 : q*B+B
            @views copyto!(C[nrng, qrng], PC[:, qrng])
            @views transpose!(C[qrng, nrng], PC[:, qrng])
        end
        @views copyto!(C[nrng, nrng], nd)
        o = new_b                                 # new block is now logical 0
    end
    return o
end

# Compat wrapper: canonical-in, canonical-out (de-rotates into a fresh matrix).
# The solvers below avoid the de-rotation by packing through lmap directly.
function _applyH_period!(ws::V9Workspace, eng, src::Matrix{Float64})
    src === ws.C || copyto!(ws.C, src)
    o = _applyH_period_ring!(ws, eng)
    _fill_lmap!(ws.lmap, o, eng.BSIZE, eng.r+1)
    W=eng.W; out=Matrix{Float64}(undef, W, W)
    @inbounds for j in 1:W, i in 1:W
        out[i,j] = ws.C[ws.lmap[i], ws.lmap[j]]
    end
    # ws.C is left rotated by the ring pass; restore canonical layout so a
    # repeated in-place call (src === ws.C) starts from valid data again
    src === ws.C && copyto!(ws.C, out)
    out
end

# column-major upper-triangle vech index list (cache-friendly unpack: the
# inner i-loop walks straight down a column of C)
function _vech_idx(W)
    idx=Vector{Tuple{Int,Int}}(undef, W*(W+1)÷2)
    k=0
    @inbounds for j in 1:W, i in 1:j
        k+=1; idx[k]=(i,j)
    end
    idx
end

# pack the rotated ring buffer into canonical vech order, iterating over the
# PHYSICAL upper triangle (sequential, cache-friendly reads of C; the vech
# position is recovered through the inverse column map — scattered writes are
# far cheaper than the scattered reads of a logical-order gather)
function _pack_ring(C, imap, Nv)
    v=zeros(Nv); W=length(imap)
    @inbounds for pj in 1:W
        j0=imap[pj]
        for pi in 1:pj
            i0=imap[pi]
            i,j = i0<=j0 ? (i0,j0) : (j0,i0)
            v[(j*(j-1))>>1 + i] = C[pi,pj]
        end
    end
    v
end
_inv_lmap(lmap) = (imap=similar(lmap); @inbounds for c in eachindex(lmap); imap[lmap[c]]=c; end; imap)

function rho_H_krylov_v9m(eng; tol=1e-11, krylovdim=30)
    haskey(eng, :ops) || return _rho_H_krylov_v9m_ref(eng; tol=tol, krylovdim=krylovdim)
    W=eng.W; B=eng.BSIZE; nblk=eng.r+1
    idx=_vech_idx(W); Nv=length(idx)
    ws=V9Workspace(eng)
    lmap=_fill_lmap!(ws.lmap, mod(-eng.p, nblk), B, nblk)   # same offset every matvec
    imap=_inv_lmap(lmap)
    C=ws.C
    unpack!(v)=(@inbounds for k in 1:Nv;(i,j)=idx[k];C[i,j]=v[k];C[j,i]=v[k];end)
    fill!(C,0.0); _applyH_period_ring!(ws,eng); D=_pack_ring(C,imap,Nv)   # affine offset H(0)
    function op(v)
        unpack!(v); _applyH_period_ring!(ws,eng); _pack_ring(C,imap,Nv) .- D
    end
    x0=zeros(Nv); @inbounds for k in 1:Nv;(i,j)=idx[k]; i==j && (x0[k]=1.0); end
    # eager: stop as soon as the dominant eigenpair meets tol instead of always
    # building the full krylovdim-dimensional basis (halves the matvec count)
    vals,_,_=KrylovKit.eigsolve(op,x0,1,:LM;tol=tol,krylovdim=min(krylovdim,Nv),
                                maxiter=300,eager=true)
    maximum(abs.(vals))
end

# reference path (used only if an engine lacks precomputed ops)
function _rho_H_krylov_v9m_ref(eng; tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    vec2sym(v)=(C=zeros(W,W); @inbounds for k in 1:Nv;(i,j)=idx[k];C[i,j]=v[k];C[j,i]=v[k];end; C)
    sym2vec(C)=(v=zeros(Nv); @inbounds for k in 1:Nv;(i,j)=idx[k];v[k]=C[i,j];end; v)
    D=sym2vec(_applyH(eng, zeros(W,W)))
    op(v)= sym2vec(_applyH(eng, vec2sym(v))) .- D
    x0=sym2vec(Matrix{Float64}(I,W,W))
    vals,_,_=KrylovKit.eigsolve(op,x0,1,:LM;tol=tol,krylovdim=min(krylovdim,Nv),maxiter=300)
    maximum(abs.(vals))
end

function fixPoint_v9m(eng; tol=1e-11, krylovdim=30)
    haskey(eng, :ops) || return _fixPoint_v9m_ref(eng; tol=tol, krylovdim=krylovdim)
    W=eng.W; B=eng.BSIZE; nblk=eng.r+1
    idx=_vech_idx(W); Nv=length(idx)
    ws=V9Workspace(eng)
    lmap=_fill_lmap!(ws.lmap, mod(-eng.p, nblk), B, nblk)
    imap=_inv_lmap(lmap)
    C=ws.C
    unpack!(v)=(@inbounds for k in 1:Nv;(i,j)=idx[k];C[i,j]=v[k];C[j,i]=v[k];end)
    fill!(C,0.0); _applyH_period_ring!(ws,eng); dvec=_pack_ring(C,imap,Nv)
    Hlin(v)=(unpack!(v); _applyH_period_ring!(ws,eng); _pack_ring(C,imap,Nv) .- dvec)
    sol,info=KrylovKit.linsolve(v->v .- Hlin(v), dvec, dvec; tol=tol,
                                krylovdim=min(krylovdim,Nv), maxiter=300)
    info.converged==0 && @warn "fixPoint_v9m: not fully converged" info
    Cout=zeros(W,W)
    @inbounds for k in 1:Nv; (i,j)=idx[k]; Cout[i,j]=sol[k]; Cout[j,i]=sol[k]; end
    Cout
end

# reference fixpoint path (used only if an engine lacks precomputed ops)
function _fixPoint_v9m_ref(eng; tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    vec2sym(v)=(C=zeros(W,W); @inbounds for k in 1:Nv;(i,j)=idx[k];C[i,j]=v[k];C[j,i]=v[k];end; C)
    sym2vec(C)=(v=zeros(Nv); @inbounds for k in 1:Nv;(i,j)=idx[k];v[k]=C[i,j];end; v)
    D=_applyH(eng, zeros(W,W)); dvec=sym2vec(D)
    Hlin(v)=sym2vec(_applyH(eng, vec2sym(v))) .- dvec
    sol,info=KrylovKit.linsolve(v->v .- Hlin(v), dvec, dvec; tol=tol,
                                krylovdim=min(krylovdim,Nv), maxiter=300)
    info.converged==0 && @warn "fixPoint_v9m: not fully converged" info
    vec2sym(sol)
end

# ------------------------------------------------------------------ cov_colloc_vT
# =============================================================================
# cov_colloc_vT.jl — TIME-VARYING delay τ(t): fractional-limit integrated-
# history engine (generalizes the v9 J-DOF design to a smooth, T-periodic,
# non-vanishing delay; single delay, single Wiener channel, β ≡ 0).
#
# Reading map. With ξ(t) = t − τ(t) strictly increasing (ξ′ = 1 − τ′ > 0),
# the delayed-drift integral of reading step n, stage i is, after u = ξ(s),
#     ∫_{t_n}^{t_n+c_i h} B(s)·x(s−τ(s)) ds = ∫_{ξ(t_n)}^{ξ(t_n+c_i h)} B̃(u)·x(u) du,
#     B̃(u) = B(ξ⁻¹(u)) / ξ′(ξ⁻¹(u))            — a single GLOBAL weight function.
# All reading limits {ξ(t_n + θ h) : θ ∈ {0, c_1..c_S, 1}} are known a priori,
# so each window block [t_m, t_m+h] stores CUMULATIVE weighted-history DOFs
#     G_k = ∫_{t_m}^{v_k} B̃(u)·x(u) du
# at the sorted reading-image breakpoints v_k that fall inside it (v_last =
# t_m+h always; the per-block breakpoint pattern is p-periodic). Every reading
# integral is then an EXACT ±1 signed sum of stored G's:
#     F(q_hi) − F(q_lo) = G^{(j_hi)}(q_hi) − G^{(j_lo)}(q_lo) + Σ_j G^{(j)}(end),
# summing full-block ends over j_lo ≤ j < j_hi — no interpolation anywhere, so
# rough (Wiener-driven) delayed reads carry no order cap, exactly as in v8/v9.
# Constant τ = r·h reduces to the v9 construction verbatim (breakpoints =
# stage nodes, single-selector reads into block r−1).
#
# Stage equations (reading step n):  Y_i = x_n + h Σ_j a_ij A_j Y_j + J̃_i,
# x_e = x_n + h Σ_j b_j A_j Y_j + J̃_e, with J̃ the signed G-sums above.
#
# Noise increment ΔB: node/G blocks from the same causal kernel machinery as
# v9 (η ≡ 0 at window nodes; delayed reads of η vanish — needs only τ(t) ≥ h,
# not alignment), with Bf(θh) → B̃(t_n + θh) and the per-step breakpoint list
# replacing {c, 1}. Padding: blocks whose breakpoint count is below the global
# max NJ duplicate the full-block DOF (readers never reference pads).
#
# Block layout (BSIZE = (NJ+1)d): [x_e; G_1..G_NJ].
# Requires: τ(t) ≥ h, ξ′ ≥ 0.1, τ T-periodic, single delay, β ≡ 0.
# =============================================================================

struct ProbT
    d::Int; T::Float64
    τf::Function                 # t ↦ τ(t)  (smooth, T-periodic, ≥ h)
    τmin::Float64; τmax::Float64 # grid-sampled bounds over one period
    A::Function; B::Function; α::Function; β::Function
    σ::Function
end
ProbT(d,T,τf,τmin,τmax,A,B,α,β) = ProbT(d,T,τf,τmin,τmax,A,B,α,β, t->zeros(d,1))

function _no_delay_noise(pb::ProbT; nt=64)
    for k in 0:nt-1
        maximum(abs, pb.β((k+0.5)/nt * pb.T)) > 1e-14 && return false
    end
    true
end

# central-difference τ′ (build-time only; δ at the FD sweet spot)
_dtau(τf, t; δ=6.0e-6) = (τf(t+δ) - τf(t-δ)) / (2δ)

# ξ⁻¹(u): solve w − τ(w) = u by bisection on [u+τmin, u+τmax] (ξ′ > 0 ⇒ the
# bracket function is increasing; bounds padded 1% against sampling slack).
function _xi_inv(pb::ProbT, u)
    pad = 0.01*max(pb.τmax - pb.τmin, 1e-8*pb.τmax)
    lo = u + pb.τmin - pad; hi = u + pb.τmax + pad
    flo = lo - pb.τf(lo) - u
    fhi = hi - pb.τf(hi) - u
    (flo <= 0.0 && fhi >= 0.0) ||
        error("ξ⁻¹ bracket failed at u=$u (τ bounds too tight — is τ(t) T-periodic?)")
    for _ in 1:80
        mid = 0.5*(lo+hi)
        f = mid - pb.τf(mid) - u
        if f < 0.0; lo = mid; else; hi = mid; end
        hi - lo < 4*eps(abs(u) + pb.τmax + 1.0) && break
    end
    0.5*(lo+hi)
end

struct StepVT
    Pblock::Matrix{Float64}        # BSIZE×W new-block rows [x_e; G_1..G_NJ]
    Yrows::Matrix{Float64}         # Sd×W stage rows (computed, NOT persisted)
    As::Vector{Matrix{Float64}}; αs::Vector{Matrix{Float64}}
    σs::Vector{Matrix{Float64}}
    Bt::Function                   # memoized θ ∈ [0,1] ↦ B̃(t_n + θh)  (d×d)
    θbrk::Vector{Float64}          # NJ breakpoints (θ units; pads = 1.0)
    nbrk::Int                      # genuine breakpoints (θbrk[nbrk] == 1.0)
    a::Matrix{Float64}; b::Vector{Float64}; c::Vector{Float64}
    lcoef::Vector{Vector{Float64}}
    φstage::Vector{Matrix{Float64}}
    h::Float64; d::Int; S::Int; W::Int; BSIZE::Int; r::Int; NJ::Int
end

# One reading piece: window slot `lag` (0-based), G index `idx` (0 = block
# start ⇒ contributes nothing), sign ±1.
const _VTPiece = Tuple{Int,Int,Float64}

function step_vT(pb::ProbT, a, b, c, h, t_n, r_buf,
                 θbrk::Vector{Float64}, nbrk::Int,
                 readmap::Vector{Vector{_VTPiece}})
    d=pb.d; S=length(c); NJ=length(θbrk); BSIZE=(NJ+1)*d; W=(r_buf+1)*BSIZE
    As=[Matrix(pb.A(t_n+c[i]*h)) for i in 1:S]
    αs=[Matrix(pb.α(t_n+c[i]*h)) for i in 1:S]
    σs=[Matrix(pb.σ(t_n+c[i]*h)) for i in 1:S]
    # global weight B̃ on THIS block, memoized per θ (bisection runs once per node)
    Btcache=Dict{Float64,Matrix{Float64}}()
    Bt(θ) = get!(Btcache, θ) do
        u = t_n + θ*h
        w = _xi_inv(pb, u)
        ξp = 1.0 - _dtau(pb.τf, w)
        Matrix(pb.B(w)) ./ ξp
    end
    Id=Matrix{Float64}(I,d,d)
    lcoef=_lagr_coefs(c)
    xn_rng = 1:d
    Gcol(lag, idx) = lag*BSIZE + idx*d          # 0-based col offset of G_idx in slot lag
    # stage solve: (I − h a⊗A) Ystack = 1⊗x_n + J̃_del (signed G-sums)
    M=Matrix{Float64}(I,S*d,S*d)
    for i in 1:S, j in 1:S; M[(i-1)*d+1:i*d,(j-1)*d+1:j*d] .-= h*a[i,j].*As[j]; end
    Minv=inv(M)
    RHS=zeros(S*d, W)
    for i in 1:S
        RHS[(i-1)*d+1:i*d, xn_rng] .= Id
        for (lag, idx, sgn) in readmap[i]
            idx == 0 && continue
            base = Gcol(lag, idx)
            for q in 1:d; RHS[(i-1)*d+q, base+q] += sgn; end
        end
    end
    Yrows=Minv*RHS
    # endpoint row
    erow=zeros(d, W); erow[:, xn_rng] .= Id
    for (lag, idx, sgn) in readmap[S+1]
        idx == 0 && continue
        base = Gcol(lag, idx)
        for q in 1:d; erow[q, base+q] += sgn; end
    end
    for j in 1:S; erow .+= h*b[j].*(As[j]*Yrows[(j-1)*d+1:j*d, :]); end
    # continuous output K = (a⁻¹ ⊗ I)(Y − 1 x_n)/h
    Ainv=inv(a)
    Krows=zeros(S*d, W)
    for j in 1:S, m in 1:S
        Krows[(j-1)*d+1:j*d, :] .+= Ainv[j,m].*Yrows[(m-1)*d+1:m*d, :]
        Krows[(j-1)*d+1:j*d, xn_rng] .-= Ainv[j,m].*Id
    end
    Krows ./= h
    # new G rows: cumulative ∫ B̃(t_n+s)·x(t_n+s) ds at the breakpoints, by
    # per-segment Gauss quadrature on the dense output x(θh)=x_n+hΣ_j ℓint_j(θ)K_j
    Grows=zeros(NJ*d, W)
    acc=zeros(d, W)
    θprev=0.0
    for k in 1:nbrk
        θk=θbrk[k]
        if θk > θprev + 1e-14
            Wx=zeros(d,d); Wk=[zeros(d,d) for _ in 1:S]
            for (gx,gw) in zip(_G8.x, _G8.w)
                θ=θprev+(θk-θprev)*gx; wq=(θk-θprev)*h*gw
                Bs=Bt(θ)
                Wx .+= wq.*Bs
                for j in 1:S; Wk[j] .+= (wq*h*_lint(lcoef[j], θ)).*Bs; end
            end
            acc[:, xn_rng] .+= Wx
            for j in 1:S; acc .+= Wk[j]*Krows[(j-1)*d+1:j*d, :]; end
        end
        Grows[(k-1)*d+1:k*d, :] .= acc
        θprev=θk
    end
    for k in nbrk+1:NJ                          # pads duplicate the full-block DOF
        Grows[(k-1)*d+1:k*d, :] .= Grows[(nbrk-1)*d+1:nbrk*d, :]
    end
    Pblock=vcat(erow, Grows)
    RHSΦ=zeros(S*d, d); for i in 1:S; RHSΦ[(i-1)*d+1:i*d, :] .= Id; end
    Φstack=Minv*RHSΦ
    φstage=[Φstack[(k-1)*d+1:k*d, :] for k in 1:S]
    return StepVT(Pblock, Yrows, As, αs, σs, Bt, θbrk, nbrk, a, b, c, lcoef,
                  φstage, h, d, S, W, BSIZE, r_buf, NJ)
end

# helpers mirroring the v9 versions but taking a StepVT
_φ_atT(st::StepVT, θ) = begin
    Φ=Matrix{Float64}(I, st.d, st.d)
    for j in 1:st.S; Φ .+= (st.h*_lint(st.lcoef[j], θ)).*(st.As[j]*st.φstage[j]); end
    Φ
end
_Σn_atT(st::StepVT, θ, Σs, Egg) = begin
    out=zeros(st.d,st.d)
    for j in 1:st.S
        rhs = st.As[j]*Σs[j] .+ Σs[j]*st.As[j]' .+ st.αs[j]*Σs[j]*st.αs[j]' .+ Egg[j]
        out .+= (st.h*_lint(st.lcoef[j], θ)).*rhs
    end
    out
end
_ΔkerT(st::StepVT, θa, θb, Σs, Egg, φc) =
    θa<=θb ? _Σn_atT(st,θa,Σs,Egg)*(φc(θb)/φc(θa))' :
             (φc(θa)/φc(θb))*_Σn_atT(st,θb,Σs,Egg)

# ΔB for the vT block [x_e; G_1..G_NJ]: identical causal-kernel machinery to
# noise_block_v9m with Bf(θh) → B̃(t_n+θh) and {c,1} → the breakpoint list.
function noise_block_vT(st::StepVT, C)
    d=st.d; S=st.S; h=st.h; a=st.a; b=st.b; BSIZE=st.BSIZE; NJ=st.NJ
    Id=Matrix{Float64}(I,d,d)
    Egg=Vector{Matrix{Float64}}(undef,S)
    for k in 1:S
        Yk=@view st.Yrows[(k-1)*d+1:k*d, :]
        Mxx=Yk*C*Yk'
        e = st.αs[k]*Mxx*st.αs[k]' .+ st.σs[k]*st.σs[k]'
        Egg[k]=(e.+e')./2
    end
    d2=d*d
    Mop=Matrix{Float64}(I,S*d2,S*d2)
    for i in 1:S, j in 1:S
        Lj=kron(Id,st.As[j]) .+ kron(st.As[j],Id) .+ kron(st.αs[j],st.αs[j])
        Mop[(i-1)*d2+1:i*d2, (j-1)*d2+1:j*d2] .-= h*a[i,j].*Lj
    end
    rhs=zeros(S*d2)
    for j in 1:S, k in 1:S; rhs[(j-1)*d2+1:j*d2] .+= h*a[j,k].*vec(Egg[k]); end
    vΣ=Mop\rhs
    Σs=[reshape(vΣ[(k-1)*d2+1:k*d2],d,d) for k in 1:S]
    endm=zeros(d,d)
    for j in 1:S
        endm .+= h*b[j].*(st.As[j]*Σs[j] .+ Σs[j]*st.As[j]' .+ st.αs[j]*Σs[j]*st.αs[j]' .+ Egg[j])
    end
    cache=Dict{Float64,Matrix{Float64}}(); φc(θ)=get!(()->_φ_atT(st,θ),cache,θ)
    rng_G(k) = (k*d+1 : (k+1)*d)
    ΔB=zeros(BSIZE,BSIZE)
    ΔB[1:d, 1:d] .= endm
    # E[Gη_k η(x_e)ᵀ] — x_e is the endpoint node θ=1 ≥ θ_k ⇒ single segment
    for k in 1:NJ
        θa=st.θbrk[k]; acc=zeros(d,d)
        if θa > 1e-14
            for (gx,gw) in zip(_G8.x,_G8.w)
                θ=θa*gx
                acc .+= (θa*gw).*(st.Bt(θ)*_ΔkerT(st,θ,1.0,Σs,Egg,φc))
            end
        end
        V=h.*acc
        ΔB[rng_G(k), 1:d] .= V; ΔB[1:d, rng_G(k)] .= V'
    end
    # E[Gη_i Gη_jᵀ]
    for i in 1:NJ, j in 1:NJ
        j < i && continue
        θa=st.θbrk[i]; θb=st.θbrk[j]; acc=zeros(d,d)
        if θa > 1e-14 && θb > 1e-14
            for (gx,gw) in zip(_G8.x,_G8.w)
                ϑ=θb*gx; wϑ=θb*gw; Bv=st.Bt(ϑ)
                segs = ϑ<θa ? ((0.0,ϑ),(ϑ,θa)) : ((0.0,θa),)
                for (lo,hi) in segs
                    hi<=lo && continue
                    for (gx2,gw2) in zip(_G8.x,_G8.w)
                        θ=lo+(hi-lo)*gx2
                        acc .+= (wϑ*(hi-lo)*gw2).*(st.Bt(θ)*_ΔkerT(st,θ,ϑ,Σs,Egg,φc)*Bv')
                    end
                end
            end
        end
        V=(h^2).*acc
        ΔB[rng_G(i), rng_G(j)] .= V
        i != j && (ΔB[rng_G(j), rng_G(i)] .= V')
    end
    return ΔB
end

function applyH_vT(eng,C)
    Ck=copy(C)
    for st in eng.steps
        W=st.W; BSIZE=st.BSIZE; keep=W-BSIZE
        P=st.Pblock; PC=P*Ck
        newdiag=PC*P' + noise_block_vT(st,Ck)
        Cnew=similar(Ck)
        Cnew[1:BSIZE,1:BSIZE]=newdiag
        Cnew[1:BSIZE,BSIZE+1:end]=PC[:,1:keep]
        Cnew[BSIZE+1:end,1:BSIZE]=transpose(PC[:,1:keep])
        Cnew[BSIZE+1:end,BSIZE+1:end]=Ck[1:keep,1:keep]
        Ck=Cnew
    end
    return Ck
end

rho_U_vT(eng)=maximum(abs.(eigen(eng.U).values))

function build_vT(pb::ProbT, S, p; force=false, want_U::Bool=false)
    if !_no_delay_noise(pb)
        force || error("the time-varying-delay collocation engine supports no delayed " *
            "multiplicative noise (β ≡ 0); use method=ClassicalSD(q) for β ≢ 0, or " *
            "force=true to IGNORE the delayed noise (diagnostics only)")
        @warn "build_vT(force=true) on a problem with β ≢ 0: the delayed " *
              "multiplicative noise is IGNORED — diagnostics only, ρ will be wrong" maxlog=1
    end
    a,b,c=gl_tab(S); h=pb.T/p
    pb.τmin >= h*(1.0-1e-12) ||
        error("time-varying delay requires τ(t) ≥ h = T/n_steps: sampled min τ = " *
              "$(pb.τmin) < h = $h — use n_steps ≥ $(ceil(Int, pb.T/pb.τmin))")
    # smoothness/monotonicity of the reading map ξ(t)=t−τ(t), 16× oversampled
    for k in 0:16p-1
        t=(k+0.5)/(16p)*pb.T
        ξp=1.0-_dtau(pb.τf, t)
        ξp >= 0.1 || error("reading map ξ(t)=t−τ(t) must be uniformly increasing: " *
                           "ξ′($t) = $ξp < 0.1 (|τ′| too large for this engine)")
    end
    maximum(abs(pb.τf(k/64*pb.T + pb.T) - pb.τf(k/64*pb.T)) for k in 0:63) <=
        1e-9*max(pb.τmax,1.0) ||
        @warn "τ(t) does not appear T-periodic (τ(t+T) ≠ τ(t)); the period map " *
              "assumes exact T-periodicity" maxlog=1
    ξ(t)=t-pb.τf(t)
    r_buf=ceil(Int, pb.τmax/h - 1e-12) + 1
    # ---- global reading-image points q[n][i] = ξ(t_n + θoffs[i]·h), θoffs=[0;c;1]
    θoffs=vcat(0.0, c, 1.0)
    tolθ=1e-9
    # locate u on the mesh: absolute block j (covers [(j−1)h, jh]) + snapped θpos
    function locate(u)
        x=u/h
        j=floor(Int, x)+1
        θ=x-(j-1)
        if θ < tolθ
            return (j, 0.0)                     # block start
        elseif θ > 1.0-tolθ
            return (j, 1.0)                     # block end
        end
        (j, θ)
    end
    locs=[[locate(ξ((n-1)*h + θo*h)) for θo in θoffs] for n in 1:p]
    # ---- per-residue-class breakpoint lists (pattern is p-periodic)
    cls(j)=mod(j-1,p)+1
    interior=[Float64[] for _ in 1:p]
    for n in 1:p, (j,θ) in locs[n]
        (θ==0.0 || θ==1.0) && continue
        push!(interior[cls(j)], θ)
    end
    brks=Vector{Vector{Float64}}(undef,p)
    for m in 1:p
        v=sort(interior[m]); u=Float64[]
        for θ in v
            (isempty(u) || θ-u[end] > 1e-8) && push!(u, θ)
        end
        push!(u, 1.0)
        brks[m]=u
    end
    NJ=maximum(length.(brks))
    θbrks=[vcat(brks[m], fill(1.0, NJ-length(brks[m]))) for m in 1:p]
    nbrks=[length(brks[m]) for m in 1:p]
    # ---- resolve a located point to its breakpoint index in its class
    function bidx(j, θ)
        θ==0.0 && return 0
        m=cls(j)
        θ==1.0 && return nbrks[m]
        k=findfirst(x->abs(x-θ)<=1e-8, brks[m])
        k===nothing && error("internal vT bookkeeping error: breakpoint lookup failed " *
                             "(block $j, θ=$θ) — please report")
        k
    end
    # ---- reading maps: readmap[n][i], i=1..S (stages) and S+1 (endpoint)
    readmaps=Vector{Vector{Vector{_VTPiece}}}(undef,p)
    for n in 1:p
        (jlo,θlo)=locs[n][1]
        klo=bidx(jlo,θlo)
        rm=Vector{Vector{_VTPiece}}(undef,S+1)
        for i in 1:S+1
            (jhi,θhi)=locs[n][i+1]
            khi=bidx(jhi,θhi)
            pieces=_VTPiece[]
            lag(j)=(n-1)-j                       # window slot of absolute block j
            if jlo==jhi
                khi != 0  && push!(pieces,(lag(jhi), khi,  1.0))
                klo != 0  && push!(pieces,(lag(jlo), klo, -1.0))
            else
                klo != 0  && push!(pieces,(lag(jlo), klo, -1.0))
                for j in jlo:jhi-1
                    push!(pieces,(lag(j), nbrks[cls(j)], 1.0))
                end
                khi != 0  && push!(pieces,(lag(jhi), khi,  1.0))
            end
            for (lg,_,_) in pieces
                0 <= lg <= r_buf || error("internal vT bookkeeping error: window slot " *
                    "$lg out of range 0..$r_buf (n=$n, i=$i) — please report")
            end
            rm[i]=pieces
        end
        readmaps[n]=rm
    end
    # ---- per-step builds (class of the block STORED by step n is cls(n))
    steps=[step_vT(pb, a, b, c, h, (n-1)*h, r_buf, θbrks[cls(n)], nbrks[cls(n)],
                   readmaps[n]) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    eng=(steps=steps, W=W, BSIZE=BSIZE, p=p, r=r_buf, d=pb.d, NJ=NJ, engine=:vT)
    if want_U
        U=Matrix{Float64}(I,W,W)
        for st in steps
            Td=zeros(W,W); Td[1:BSIZE,:]=st.Pblock
            for k in 1:r_buf
                Td[k*BSIZE+1:(k+1)*BSIZE,(k-1)*BSIZE+1:k*BSIZE]=Matrix(I,BSIZE,BSIZE)
            end
            U=Td*U
        end
        eng=merge(eng,(U=U,))
    end
    eng
end

