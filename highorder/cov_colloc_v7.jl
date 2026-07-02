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
using LinearAlgebra, Printf, KrylovKit

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
end

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
println("cov_colloc_v7 loaded")
