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

function build_v9m(pb::Prob, S, p; force=false)
    if !force && !_no_delay_noise(pb)
        return build_v8m(pb, S, p)               # unsafe to prune ⇒ fall back to v8
    end
    a,b,c=gl_tab(S); h=pb.T/p; r=round(Int,pb.τ/h)
    abs(r*h-pb.τ) < 1e-9*max(pb.τ,1.0) || error("τ/h not integer")
    r ≥ 1 || error("need r ≥ 1")
    steps=[step_v9m(pb,a,b,c,h,(n-1)*h,r) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    U=Matrix{Float64}(I,W,W)
    for st in steps
        Td=zeros(W,W); Td[1:BSIZE,:]=st.Pblock
        for k in 1:r; Td[k*BSIZE+1:(k+1)*BSIZE,(k-1)*BSIZE+1:k*BSIZE]=Matrix(I,BSIZE,BSIZE); end
        U=Td*U
    end
    return (steps=steps,U=U,W=W,BSIZE=BSIZE,p=p,engine=:v9)
end

function applyH_v9m(eng,C)
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

# dispatch helpers: v9 eng (has :engine=:v9) vs v8 fallback (a plain NamedTuple)
_applyH(eng,C) = haskey(eng,:engine) && eng.engine==:v9 ? applyH_v9m(eng,C) : applyH_v8m(eng,C)

function rho_H_krylov_v9m(eng; tol=1e-11, krylovdim=30)
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

