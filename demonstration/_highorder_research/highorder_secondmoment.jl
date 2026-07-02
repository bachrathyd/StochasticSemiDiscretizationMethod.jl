# =============================================================================
# HIGH-ORDER second-moment stability for linear SDDEs — beating the SDM O(h³) wall.
#
# Idea (rigorous): for a LINEAR SDDE the history-segment covariance C=E[y⊗y] obeys a
# DETERMINISTIC linear Lyapunov map over one period:
#       C_period = U C Uᵀ + Q_period
# U  = the IMPLICIT high-order collocation monodromy (first_moment_phi) — order 2S.
# Q_period = noise covariance, built by Itô isometry with the noise coefficient sampled
#            on the COLLOCATION STAGE polynomial (NO present-state freeze) ⇒ also order 2S.
# ρ(C ↦ U C Uᵀ + Q) = ρ(H). Noise-off ⇒ ρ = ρ(U)² (machine precision).
#
# Construction of Q without re-deriving propagation: the stochastic collocation system is
#   L V = R y_hist + 𝒢 Ξ          (Ξ = independent per-step Wiener increments)
# so V = L⁻¹R y_hist + L⁻¹𝒢 Ξ, and Cov(V) gains  (L⁻¹𝒢) E[ΞΞᵀ] (L⁻¹𝒢)ᵀ. Restricting to
# the period-end window gives Q_period. The per-step noise injection 𝒢ₙ enters the stage
# equations exactly like the delayed term B does (same M⁻¹ collocation solve), but the
# noise coefficient g(s)=α(s)x(s)+β(s)x(s-τ) is evaluated on the STAGE polynomial → the
# present-state x(s) is de-frozen. E[ΞΞᵀ] couples to C (linear) via Itô isometry.
#
# PROOF stage: dense ρ via eigen, small p. (Scalable Krylov is a later step.)
# Self-contained (reuses the verified build_LR / step_blocks logic, ported here).
# Run:  julia --project=. demonstration/highorder_secondmoment.jl
# =============================================================================
using LinearAlgebra, SparseArrays, Printf

include(joinpath(@__DIR__,"sdde_types.jl"))   # SDDEProblem + maxdelay

# ---- GL(S) tableaux + collocation weights (ported, verified) ----
function gl_tableau(S::Int)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5]
    elseif S==2; s3=sqrt(3); return [0.25 0.25-s3/6;0.25+s3/6 0.25],[0.5,0.5],[0.5-s3/6,0.5+s3/6]
    elseif S==3; s15=sqrt(15)
        return [5/36 2/9-s15/15 5/36-s15/30;5/36+s15/24 2/9 5/36-s15/24;5/36+s15/30 2/9+s15/15 5/36],[5/18,4/9,5/18],[0.5-s15/10,0.5,0.5+s15/10]
    else; error("S=1,2,3"); end
end
function colloc_weights(c::Vector{Float64}, θ::Float64)
    nodes=vcat(0.0,c,1.0); n=length(nodes); w=zeros(n)
    for i in 1:n
        wi=1.0; for j in 1:n; i!=j && (wi*=(θ-nodes[j])/(nodes[i]-nodes[j])); end; w[i]=wi
    end
    return w
end

# ---- per-step deterministic blocks (verified port) + the per-step noise injection ----
# Returns, in addition to Mprop/deldata, the data needed to inject stochastic forcing:
#   Minv (SD×SD), Astage, b, c, h, t_n  so the noise RHS can be solved like the delay RHS.
struct StepData
    Mprop::Matrix{Float64}                                   # BSIZE×d
    deldata::Vector{Vector{Tuple{Matrix{Float64},Int,Vector{Float64}}}}
    Minv::Matrix{Float64}; Astage::Vector{Matrix{Float64}}
    BSIZE::Int
end

function step_data(prob::SDDEProblem,a,b,c,h,t_n,t_start,r)
    d=prob.d; S=length(c); BSIZE=(S+1)*d; SD=S*d
    Astage=[prob.A(t_n+c[i]*h) for i in 1:S]
    M=Matrix{Float64}(I,SD,SD)
    for i in 1:S,j in 1:S,di in 1:d,dj in 1:d
        M[(i-1)*d+di,(j-1)*d+dj]-=h*a[i,j]*Astage[j][di,dj]
    end
    Minv=inv(M)
    RHSy=zeros(SD,d); for i in 1:S,di in 1:d; RHSy[(i-1)*d+di,di]=1.0; end
    Yy=Minv*RHSy; ynext=Matrix{Float64}(I,d,d)
    for j in 1:S; ynext+=h*b[j]*(Astage[j]*Yy[(j-1)*d+1:j*d,:]); end
    Mprop=vcat(ynext,Yy)
    deldata=Vector{Tuple{Matrix{Float64},Int,Vector{Float64}}}[]
    for (k,(τf,Bf)) in enumerate(prob.delays)
        ps=Tuple{Matrix{Float64},Int,Vector{Float64}}[]
        for st in 1:S
            tst=t_n+c[st]*h; Bst=Bf(tst)
            RHSd=zeros(SD,d); for i in 1:S,di in 1:d,dj in 1:d; RHSd[(i-1)*d+di,dj]=h*a[i,st]*Bst[di,dj]; end
            Yd=Minv*RHSd; ynd=zeros(d,d)
            for j in 1:S; term=Astage[j]*Yd[(j-1)*d+1:j*d,:]; j==st && (term+=Bst); ynd+=h*b[j]*term; end
            Md=vcat(ynd,Yd); τval=τf(tst); rel=(tst-τval-t_start)/h+r+1; mi=floor(Int,rel)
            push!(ps,(Md,mi,colloc_weights(c,rel-mi)))
        end
        push!(deldata,ps)
    end
    return StepData(Mprop,deldata,Minv,Astage,BSIZE)
end

# ---- L,R assembly (verified port) ----
function build_LR(prob::SDDEProblem,a,b,c,p,h,t_start,r,steps::Vector{StepData})
    d=prob.d; S=length(c); BSIZE=(S+1)*d
    IL=Int[];JL=Int[];VL=Float64[]; IR=Int[];JR=Int[];VR=Float64[]
    for n in 1:p
        roff=(n-1)*BSIZE; sd=steps[n]; Mprop=sd.Mprop
        for i in 1:BSIZE; push!(IL,roff+i);push!(JL,roff+i);push!(VL,1.0); end
        for di in 1:d, rb in 1:BSIZE
            v=-Mprop[rb,di]
            if v!=0
                if n==1; push!(IR,roff+rb);push!(JR,r*BSIZE+di);push!(VR,v)
                else; push!(IL,roff+rb);push!(JL,(n-2)*BSIZE+di);push!(VL,v); end
            end
        end
        for ps in sd.deldata, (Md,mi,w) in ps
            bx=mi-(r+1)
            for dj in 1:d, rb in 1:BSIZE
                val=-Md[rb,dj]*w[1]
                if val!=0
                    if bx<=0; push!(IR,roff+rb);push!(JR,(bx+r)*BSIZE+dj);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(bx-1)*BSIZE+dj);push!(VL,val); end
                end
            end
            be=(mi+1)-(r+1)
            for ss in 1:S, dj in 1:d, rb in 1:BSIZE
                val=-Md[rb,dj]*w[ss+1]
                if val!=0
                    col=d+(ss-1)*d+dj
                    if be<=0; push!(IR,roff+rb);push!(JR,(be+r)*BSIZE+col);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(be-1)*BSIZE+col);push!(VL,val); end
                end
            end
            for dj in 1:d, rb in 1:BSIZE
                val=-Md[rb,dj]*w[S+2]
                if val!=0
                    if be<=0; push!(IR,roff+rb);push!(JR,(be+r)*BSIZE+dj);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(be-1)*BSIZE+dj);push!(VL,val); end
                end
            end
        end
    end
    L=sparse(IL,JL,VL,p*BSIZE,p*BSIZE); R=sparse(IR,JR,VR,p*BSIZE,(r+1)*BSIZE)
    return L,R,BSIZE
end

# =============================================================================
# The implicit high-order monodromy U (window→window), built from L,R (verified design).
# Returns U (W×W), plus everything needed to build Q: Lf=lu(L), R, steps, p,r,BSIZE,h,c,b.
# =============================================================================
struct PeriodOp
    U::Matrix{Float64}
    Lf                      # lu factorization of L
    R::SparseMatrixCSC{Float64,Int}
    steps::Vector{StepData}
    p::Int; r::Int; BSIZE::Int; d::Int; S::Int; h::Float64
    a::Matrix{Float64}; b::Vector{Float64}; c::Vector{Float64}
    prob::SDDEProblem
end

function build_period_op(prob::SDDEProblem, S, p)
    a,b,c = gl_tableau(S); h=prob.T/p
    ts=range(0,prob.T,length=p+1); r=max(round(Int,maxdelay(prob,ts)/h),1)
    steps=[step_data(prob,a,b,c,h,(n-1)*h,0.0,r) for n in 1:p]
    L,R,BSIZE=build_LR(prob,a,b,c,p,h,0.0,r,steps)
    Lf=lu(Matrix(L)); Rm=Matrix(R); W=(r+1)*BSIZE
    # U via base_sweep emulation (verified): load window→hist, solve, read period-end window
    U=zeros(W,W)
    for k in 1:W
        x=zeros(W); x[k]=1.0; vh=zeros((r+1)*BSIZE)
        for i in 0:r; vh[(r-i)*BSIZE+1:(r-i+1)*BSIZE]=x[i*BSIZE+1:(i+1)*BSIZE]; end
        vper=Lf\(Rm*vh); y=zeros(W)
        for i in 0:r
            kk=p-i
            if kk>=1; y[i*BSIZE+1:(i+1)*BSIZE]=vper[(kk-1)*BSIZE+1:kk*BSIZE]
            else; y[i*BSIZE+1:(i+1)*BSIZE]=vh[(kk+r)*BSIZE+1:(kk+r+1)*BSIZE]; end
        end
        U[:,k]=y
    end
    return PeriodOp(U,Lf,R,steps,p,r,BSIZE,prob.d,S,h,a,b,c,prob)
end

# ρ(U): dominant deterministic Floquet multiplier (order 2S)
rho_U(op::PeriodOp) = maximum(abs.(eigen(op.U).values))

# Noise-off Lyapunov map ρ: should equal ρ(U)². (The make-or-break identity, per dimension.)
function rho_noisefree(op::PeriodOp)
    W=size(op.U,1); U=op.U
    # operator C ↦ U C Uᵀ on symmetric W×W → dominant eigenvalue via dense build (small p)
    # vech basis
    idx=Tuple{Int,Int}[]; for i in 1:W, j in i:W; push!(idx,(i,j)); end
    Nv=length(idx); H=zeros(Nv,Nv)
    pos=Dict(idx[k]=>k for k in 1:Nv)
    for k in 1:Nv
        (i,j)=idx[k]; C=zeros(W,W); C[i,j]=1.0; C[j,i]=1.0
        Cn=U*C*U'
        for m in 1:Nv; (a2,b2)=idx[m]; H[m,k]=Cn[a2,b2]; end
    end
    return maximum(abs.(eigen(H).values))
end

# =============================================================================
# quick self-test: noise-off identity ρ(UCUᵀ)=ρ(U)² for d=1,2,3
# =============================================================================
function _hayes_prob()
    SDDEProblem(1,1.0, t->reshape([-1.0],1,1), [(t->1.0,t->reshape([-0.4],1,1))],
        [(t->reshape([0.3],1,1),[t->reshape([0.2],1,1)],t->[0.0])])
end
function _osc2_prob()   # d=2 with present-state α and cross-coupling
    SDDEProblem(2,2π, t->[0.0 1.0; -4.0 -0.3], [(t->2π,t->[0.0 0.0; -0.5 0.0])],
        [(t->[0.0 0.0; 0.1 0.15],[t->[0.0 0.0; 0.1 0.0]],t->[0.0,0.0])])
end
function _sys3_prob()   # d=3
    A=[-2.0 0.3 0.0; 0.1 -2.0 0.2; 0.0 0.1 -2.0]; B=0.2*[0 0 0;1 0 0;0 1 0.0]
    SDDEProblem(3,1.0, t->A, [(t->1.0,t->B)], [(t->0.1*Matrix(I,3,3),[t->zeros(3,3)],t->[0.0,0,0])])
end

# =============================================================================
# NOISE COVARIANCE Q — the de-frozen Itô-isometry contribution (where order is won).
#
# Per step n, per noise source w, per Gauss node i (time sᵢ=tₙ+cᵢh, weight bᵢh):
#   the noise increment is  Ĝₙᵢ · (g(sᵢ) √(bᵢh))   where g(sᵢ)=α(sᵢ)x(sᵢ)+β(sᵢ)x(sᵢ-τ).
#   Ĝₙᵢ : R^d → BSIZE block, the collocation response to a forcing placed at stage i
#         (built via Minv, identical to how the delay term propagates).
# The increment lands in the period via L⁻¹ (implicit, high order). Its covariance is
#   (bᵢh) · Ĝprop · E[g(sᵢ)g(sᵢ)ᵀ] · Ĝpropᵀ ,  Ĝprop = (window-extract of L⁻¹ · placed-Ĝₙᵢ)
# E[g gᵀ] = α Mxx αᵀ + α Mxτ βᵀ + β Mτx αᵀ + β Mττ βᵀ, all blocks of the window covariance C
# evaluated AT sᵢ via the collocation polynomial (present x(sᵢ) from stage values → de-frozen).
#
# We assemble Q as a dense linear operator on vech(C) by applying it to basis covariances.
# =============================================================================

# Collocation response to a unit forcing of the d-vector at stage `st`: returns BSIZE×d.
# (A forcing φ at stage st means stage equations get h·a[i,st]·φ; same path as delay B=I.)
function stage_force_response(op::PeriodOp, n::Int, st::Int)
    sd=op.steps[n]; d=op.d; S=op.S; BSIZE=op.BSIZE; h=op.h; a=op.a; b=op.b
    SD=S*d; Minv=sd.Minv; Astage=sd.Astage
    RHS=zeros(SD,d)
    for i in 1:S, di in 1:d; RHS[(i-1)*d+di,di]=h*a[i,st]; end   # forcing only at stage st
    Yd=Minv*RHS
    ynd=zeros(d,d)
    for j in 1:S
        term=Astage[j]*Yd[(j-1)*d+1:j*d,:]; j==st && (term+=Matrix(I,d,d)); ynd+=h*b[j]*term
    end
    return vcat(ynd,Yd)   # BSIZE×d
end

# window-extract of L⁻¹ applied to a per-step block placed at step n's rows (BSIZE×d → W×d)
function propagate_block_to_window(op::PeriodOp, n::Int, blk::Matrix{Float64})
    p=op.p; r=op.r; BSIZE=op.BSIZE; d=op.d; W=(r+1)*BSIZE
    rhs=zeros(p*BSIZE, d)
    rhs[(n-1)*BSIZE+1:n*BSIZE, :] = blk
    V = op.Lf \ rhs                       # (p*BSIZE)×d : the period response to this forcing
    Y=zeros(W,d)
    for i in 0:r
        kk=p-i
        if kk>=1; Y[i*BSIZE+1:(i+1)*BSIZE,:]=V[(kk-1)*BSIZE+1:kk*BSIZE,:]
        end   # kk<1 would read history (no noise there) → 0
    end
    return Y                              # W×d : how a stage-forcing at step n shows up at period end
end

# Build, once, the per-(step,stage) propagated forcing operators P[n][st] :: W×d.
function build_force_props(op::PeriodOp)
    P=[Vector{Matrix{Float64}}(undef, op.S) for _ in 1:op.p]
    for n in 1:op.p, st in 1:op.S
        blk=stage_force_response(op,n,st)
        P[n][st]=propagate_block_to_window(op,n,blk)
    end
    return P
end

# =============================================================================
# Index-safe state reads: for step n, stage st, build d×W read operators that extract
# x(tₙ+c_st·h) [present] and x(tₙ+c_st·h − τ_k) [delayed] from the period-START window y.
# Built by PROBING: solve V=L⁻¹R·e for each window basis vector e and read the relevant
# block of V. This reuses the exact solver, so the indexing cannot drift.
#
# The within-period state V[block m] = the m-th produced block. The PRESENT state at
# (n,st) is stage st of block n: V row (n-1)*BSIZE + d + (st-1)*d + (1..d).
# The DELAYED state at (n,st) uses the collocation routing (mi, weights) on the window/
# produced blocks, exactly as build_LR does for B. We assemble the delayed read with the
# same routing arithmetic, but reading from the SOLVED V (so delayed blocks that fall in
# the current period are the high-order produced values, not frozen history).
# To keep it index-safe we build the FULL "window→all produced+history blocks" map once.
# =============================================================================

# Map period-start window y (W) → augmented full state Z = [history(r+1 blocks); V(p blocks)]
# stacked so that buffer block index b∈[1, p+r+1] addresses Z (1=oldest history .. newest produced).
# Returns Zmap :: ((p+r+1)*BSIZE) × W.
function build_full_state_map(op::PeriodOp)
    p=op.p; r=op.r; BSIZE=op.BSIZE; W=(r+1)*BSIZE; Rm=Matrix(op.R)
    Ntot=(p+r+1)*BSIZE
    Z=zeros(Ntot, W)
    for k in 1:W
        e=zeros(W); e[k]=1.0
        # load window→hist (buffer blocks 1..r+1, 1=oldest): same as U-build
        vh=zeros((r+1)*BSIZE)
        for i in 0:r; vh[(r-i)*BSIZE+1:(r-i+1)*BSIZE]=e[i*BSIZE+1:(i+1)*BSIZE]; end
        V=op.Lf\(Rm*vh)
        z=zeros(Ntot)
        z[1:(r+1)*BSIZE]=vh                              # history blocks (buffer 1..r+1)
        z[(r+1)*BSIZE+1:end]=V                           # produced blocks (buffer r+2..p+r+1)
        Z[:,k]=z
    end
    return Z
end

# present read (d×W): stage st of produced block n (buffer index n+r+1), stage offset.
function present_read(op::PeriodOp, Z::Matrix{Float64}, n::Int, st::Int)
    BSIZE=op.BSIZE; d=op.d
    base=( (n+r_of(op)) )*BSIZE          # produced block n is buffer (n+r+1) → 0-based (n+r)
    rows=base + d + (st-1)*d .+ (1:d)    # stage st DOFs within the block (endpoint=1:d, stages after)
    return Z[rows, :]
end
r_of(op::PeriodOp)=op.r

# delayed read (d×W): x(tₙ+c_st h − τ_k) via collocation routing on buffer blocks of Z.
function delayed_read(op::PeriodOp, Z::Matrix{Float64}, n::Int, st::Int, k::Int)
    d=op.d; S=op.S; BSIZE=op.BSIZE; r=op.r; h=op.h
    τf=op.prob.delays[k][1]; tst=(n-1)*h+op.c[st]*h
    τval=τf(tst); rel=(tst-τval-0.0)/h + r + 1; mi=floor(Int,rel); w=colloc_weights(op.c,rel-mi)
    # buffer block mi: x-part weight w[1]; block mi+1: stages w[2..S+1] + endpoint w[S+2].
    Rd=zeros(d,size(Z,2))
    if 1<=mi<=size(Z,1)÷BSIZE
        Rd .+= w[1] .* Z[(mi-1)*BSIZE .+ (1:d), :]                 # endpoint (x-part) of block mi
    end
    if 1<=mi+1<=size(Z,1)÷BSIZE
        Rd .+= w[S+2] .* Z[mi*BSIZE .+ (1:d), :]                   # endpoint of block mi+1
        for ss in 1:S
            Rd .+= w[ss+1] .* Z[mi*BSIZE + d + (ss-1)*d .+ (1:d), :]  # stages of block mi+1
        end
    end
    return Rd
end

# =============================================================================
# FULL second-moment operator 𝓗[C] = U C Uᵀ + Q[C], and its dominant eigenvalue (dense).
# Precompute: force-props P[n][st] (W×d), full-state map Z, and per-(n,st,w) noise-coef
# read operators G_read (d×W). Then 𝓗 is assembled as a dense Nv×Nv matrix on vech(C).
# =============================================================================
struct M2Op
    op::PeriodOp
    P::Vector{Vector{Matrix{Float64}}}       # P[n][st] :: W×d (propagated stage forcing)
    Gread::Vector{Vector{Vector{Matrix{Float64}}}}  # Gread[n][st][w] :: d×W (noise coef read)
    wt::Vector{Float64}                       # Gauss weight bᵢ per stage (length S)
    W::Int
end

function build_m2op(prob::SDDEProblem, S, p)
    op=build_period_op(prob,S,p)
    W=size(op.U,1); h=op.h
    P=build_force_props(op)
    Z=build_full_state_map(op)
    nw=length(prob.noise)
    Gread=[[Vector{Matrix{Float64}}(undef,nw) for _ in 1:S] for _ in 1:op.p]
    for n in 1:op.p, st in 1:S
        tst=(n-1)*h+op.c[st]*h
        Rp=present_read(op,Z,n,st)                       # d×W
        for (w,(αf,βfs,σf)) in enumerate(prob.noise)
            α=αf(tst)
            G=α*Rp                                       # present term (DE-FROZEN: stage value)
            for (k,_) in enumerate(prob.delays)
                β=βfs[k](tst)
                all(iszero,β) && continue
                G = G .+ β*delayed_read(op,Z,n,st,k)
            end
            Gread[n][st][w]=G
        end
    end
    return M2Op(op,P,Gread,op.b,W)
end

# Apply 𝓗 to a covariance C (W×W) → W×W.  𝓗[C] = U C Uᵀ + Σ_{n,st,w}(b_st h)P G C Gᵀ Pᵀ.
function apply_H(m::M2Op, C::Matrix{Float64})
    op=m.op; U=op.U; h=op.h
    out = U*C*U'
    for n in 1:op.p, st in 1:op.S
        Pns=m.P[n][st]                       # W×d
        for w in eachindex(op.prob.noise)
            G=m.Gread[n][st][w]              # d×W
            Egg = G*C*G'                     # d×d  = E[g(sᵢ)g(sᵢ)ᵀ]
            out .+= (m.wt[st]*h) .* (Pns*Egg*Pns')
        end
    end
    return out
end

# dominant eigenvalue of 𝓗 via dense build on vech (small p proof).
function rho_H(m::M2Op)
    W=m.W; idx=Tuple{Int,Int}[]; for i in 1:W, j in i:W; push!(idx,(i,j)); end
    Nv=length(idx); H=zeros(Nv,Nv)
    for k in 1:Nv
        (i,j)=idx[k]; C=zeros(W,W); C[i,j]=1.0; C[j,i]=1.0
        Cn=apply_H(m,C)
        for mrow in 1:Nv; (a2,b2)=idx[mrow]; H[mrow,k]=Cn[a2,b2]; end
    end
    return maximum(abs.(eigen(H).values))
end

# ---- test problems for the ladder ----
_present_noise_prob(α,a) = SDDEProblem(1,1.0, t->reshape([a],1,1),
    [(t->1.0, t->reshape([0.0],1,1))],                 # dummy delay, B=0 (no delay coupling)
    [(t->reshape([α],1,1),[t->reshape([0.0],1,1)],t->[0.0])])   # present-only mult. noise

if abspath(PROGRAM_FILE)==@__FILE__
    println("\n=== STEP 1: noise-off identity ρ(UCUᵀ)=ρ(U)² (make-or-break) ===")
    for (nm,prob) in [("d1 Hayes",_hayes_prob()),("d2 osc",_osc2_prob()),("d3 sys",_sys3_prob())]
        for S in [2,3]
            op=build_period_op(prob,S,8)
            ru=rho_U(op); r2=rho_noisefree(op)
            @printf("  %-9s S=%d p=8:  ρ(U)²=%.12f  ρ(UCUᵀ)=%.12f  diff=%.1e  %s\n",
                    nm,S,ru^2,r2,abs(r2-ru^2), abs(r2-ru^2)<1e-9 ? "OK" : "*** FAIL ***")
        end
    end

    println("\n=== STEP 2 LITMUS: dx=αx dW (no delay) → exp(α²T), expect GL2 O(h⁴), GL3 O(h⁶) ===")
    a_t=-0.7; α=0.5; T=1.0
    exact=exp((2a_t+α^2)*T)
    @printf("  exact exp((2a+α²)T) = %.12f\n", exact)
    for S in [1,2,3]
        @printf("  GL%d:\n",S); prev=nothing
        for p in [4,8,16,32]
            m=build_m2op(_present_noise_prob(α,a_t),S,p)
            ρ=rho_H(m); err=abs(ρ-exact)
            rate = prev===nothing ? NaN : log2(prev/err)
            @printf("    p=%2d ρ=%.11f err=%.2e rate=%.2f\n",p,ρ,err,rate); prev=err
        end
    end
end
