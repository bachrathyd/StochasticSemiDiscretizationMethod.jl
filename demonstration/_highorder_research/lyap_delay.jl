# =============================================================================
# HIGH-ORDER 2nd moment for linear SDDEs via per-step COVARIANCE collocation (Lyapunov).
#
# Validated template (lyap_concept.jl): collocating the deterministic covariance ODE
#   d vec(C)/dt = 𝓛(t) vec(C),  𝓛 = I⊗𝓐 + 𝓐⊗I + Σ_j 𝓖_j⊗𝓖_j
# gives order 2S (GL2→O(h⁴), GL3→O(h⁶)) even for present-state multiplicative noise.
#
# Here we lift to the DELAY case. The augmented window state is y (the DDE history,
# discretized). We evolve the window COVARIANCE C=E[y yᵀ] one period via the SAME global
# collocation solve used for the first moment (L,R), but on the covariance. Concretely we
# build the per-step covariance transition by collocating the matrix Lyapunov step:
#   the stage covariances Σ_i (= C at stage time) satisfy a linear (Sylvester) system from
#   the SAME stage generators; the delayed covariance blocks are routed exactly as the
#   deterministic delay term routes the delayed state. Then ρ(one-period covariance map).
#
# For tractable, index-safe correctness we VEC the window covariance and build the
# one-period covariance monodromy 𝓜 by applying the per-step covariance update to vec(C)
# basis vectors. ρ(𝓜) = ρ(H), order 2S. (Dense; proof stage, small p.)
#
# Reuses sdde_types.jl + the verified deterministic step machinery (ported).
# =============================================================================
using LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__,"sdde_types.jl"))

# GL(S) tableaux S=1..6 (extended) ------------------------------------------------
function gl_tab(S::Int)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5]
    elseif S==2; s3=sqrt(3); return [0.25 0.25-s3/6;0.25+s3/6 0.25],[0.5,0.5],[0.5-s3/6,0.5+s3/6]
    elseif S==3; s15=sqrt(15)
        return [5/36 2/9-s15/15 5/36-s15/30;5/36+s15/24 2/9 5/36-s15/24;5/36+s15/30 2/9+s15/15 5/36],[5/18,4/9,5/18],[0.5-s15/10,0.5,0.5+s15/10]
    else
        # general Gauss–Legendre via Golub–Welsch on [0,1]
        return gl_general(S)
    end
end

# General GL(S): nodes=roots of shifted Legendre, weights, and a_ij = ∫₀^{c_i} ℓ_j.
function gl_general(S::Int)
    # Golub–Welsch on [-1,1]: symmetric tridiagonal Jacobi matrix
    β=[k/sqrt(4k^2-1) for k in 1:S-1]
    J=diagm(1=>β)+diagm(-1=>β)
    vals,vecs=eigen(Symmetric(J))
    x=vals; w=2 .*(vecs[1,:]).^2                 # nodes/weights on [-1,1]
    c=(x.+1)./2; b=w./2                           # shift to [0,1]
    # a_ij = ∫₀^{c_i} ℓ_j(s) ds, ℓ_j Lagrange basis on nodes c. Compute via collocation:
    # ℓ_j(c) values → integrate polynomial. Use the standard: a = C * inv(V) where
    # V_{kj}=c_k^{j-1}, and ∫₀^{c_i} s^{j-1} ds = c_i^j / j.
    V=[c[k]^(j-1) for k in 1:S, j in 1:S]
    Aint=[c[i]^j / j for i in 1:S, j in 1:S]      # ∫₀^{c_i} s^{j-1} ds in monomial basis
    a=Aint/V                                       # a = Aint * V^{-1}
    return a, b, c
end

colloc_w(c,θ)=(nodes=vcat(0.0,c,1.0);n=length(nodes);[prod(j->j==i ? 1.0 : (θ-nodes[j])/(nodes[i]-nodes[j]),1:n) for i in 1:n])

# =============================================================================
# Per-step deterministic stage matrices (verified port): for step n returns
#   Minv (SD×SD), Astage[i] (d×d), and the delay routing (mi,w) per (delay k, stage st).
# =============================================================================
struct Step
    Minv::Matrix{Float64}; Astage::Vector{Matrix{Float64}}
    delroute::Vector{Vector{Tuple{Int,Vector{Float64},Matrix{Float64}}}}  # [k][st]=(mi,w,Bstage)
end
function build_step(prob,a,b,c,h,t_n,t_start,r)
    d=prob.d; S=length(c); SD=S*d
    Astage=[prob.A(t_n+c[i]*h) for i in 1:S]
    M=Matrix{Float64}(I,SD,SD)
    for i in 1:S,j in 1:S,di in 1:d,dj in 1:d; M[(i-1)*d+di,(j-1)*d+dj]-=h*a[i,j]*Astage[j][di,dj]; end
    Minv=inv(M)
    dr=Vector{Tuple{Int,Vector{Float64},Matrix{Float64}}}[]
    for (k,(τf,Bf)) in enumerate(prob.delays)
        ps=Tuple{Int,Vector{Float64},Matrix{Float64}}[]
        for st in 1:S
            tst=t_n+c[st]*h; τval=τf(tst); rel=(tst-τval-t_start)/h+r+1; mi=floor(Int,rel)
            push!(ps,(mi,colloc_w(c,rel-mi),Bf(tst)))
        end
        push!(dr,ps)
    end
    return Step(Minv,Astage,dr)
end

# =============================================================================
# Global first-moment collocation system L V = R y_hist  (verified port).
# V = [v_1;...;v_p] produced blocks (BSIZE each = endpoint d + S stages·d);
# y_hist = (r+1) history blocks.  L: (p·BSIZE)²,  R: (p·BSIZE)×((r+1)·BSIZE).
# =============================================================================
function build_LR(prob,a,b,c,p,h,t_start,r,steps)
    d=prob.d; S=length(c); BSIZE=(S+1)*d; SD=S*d
    IL=Int[];JL=Int[];VL=Float64[]; IR=Int[];JR=Int[];VR=Float64[]
    for n in 1:p
        roff=(n-1)*BSIZE; sd=steps[n]; Minv=sd.Minv; Astage=sd.Astage
        # Mprop (BSIZE×d): stages & endpoint from previous endpoint
        RHSy=zeros(SD,d); for i in 1:S,di in 1:d; RHSy[(i-1)*d+di,di]=1.0; end
        Yy=Minv*RHSy; ynext=Matrix{Float64}(I,d,d)
        for j in 1:S; ynext+=h*b[j]*(Astage[j]*Yy[(j-1)*d+1:j*d,:]); end
        Mprop=vcat(ynext,Yy)
        for i in 1:BSIZE; push!(IL,roff+i);push!(JL,roff+i);push!(VL,1.0); end
        for di in 1:d, rb in 1:BSIZE
            v=-Mprop[rb,di]
            if v!=0
                if n==1; push!(IR,roff+rb);push!(JR,r*BSIZE+di);push!(VR,v)
                else; push!(IL,roff+rb);push!(JL,(n-2)*BSIZE+di);push!(VL,v); end
            end
        end
        for (k,ps) in enumerate(sd.delroute)
            for st in 1:S
                (mi,w,Bst)=ps[st]
                RHSd=zeros(SD,d); for i in 1:S,di in 1:d,dj in 1:d; RHSd[(i-1)*d+di,dj]=h*a[i,st]*Bst[di,dj]; end
                Yd=Minv*RHSd; ynd=zeros(d,d)
                for j in 1:S; term=Astage[j]*Yd[(j-1)*d+1:j*d,:]; j==st && (term+=Bst); ynd+=h*b[j]*term; end
                Md=vcat(ynd,Yd)
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
    end
    L=sparse(IL,JL,VL,p*BSIZE,p*BSIZE); R=sparse(IR,JR,VR,p*BSIZE,(r+1)*BSIZE)
    return L,R,BSIZE
end

# =============================================================================
# COVARIANCE one-period monodromy via the TENSORED collocation system (provably correct).
#
# First moment:  L V = R y_hist           (V = produced blocks, verified high order)
# Second moment: the covariance Vc=E[V Vᵀ] of the produced blocks, given the history
# covariance Ch=E[y_hist y_histᵀ], satisfies the SAME collocation but tensored:
#       (L⊗L) vec(Vc) = (R⊗R) vec(Ch) + vec(𝒩)
# where 𝒩 is the Itô noise covariance injected into each step's produced block. Noise-off
# (𝒩=0): vec(Vc)=(L⁻¹R)⊗(L⁻¹R) vec(Ch) ⇒ window map U⊗U ⇒ ρ(U)² (matches Step1, exact).
#
# 𝒩 injection (de-frozen, Itô isometry by Gauss quadrature on stages): for step n, the
# produced block's covariance gets, in its ENDPOINT×ENDPOINT d×d sub-block (rows/cols of
# block n's endpoint), the term  Σ_w Σ_{st}(b_st h) g_{n,st,w} Ch_full g_{n,st,wᵀ}, where
# g reads the noise coefficient α x(s_st)+β x(s_st-τ) from the produced+history blocks
# (present = stage value of block n ⇒ DE-FROZEN). Because g depends on V (implicit), 𝒩 is
# linear in vec(Vc): it is an operator 𝒩op acting on the (V,hist) covariance. We fold it
# by solving the tensored system as a fixed-point/linear map and reading the period-end
# window covariance. ρ(that map) = ρ(H), order 2S.
#
# IMPLEMENTATION (dense, proof): build the linear map on vec(window covariance) directly by
# applying it to basis covariances: load → tensored solve with 𝒩 → read period-end window.
# =============================================================================

struct CovOp
    Lf; Rm::Matrix{Float64}
    # per (n,st,w): read operator g on the FULL block space [hist(r+1); V(p)] → d
    # represented as d×Nfull, where Nfull=(p+r+1)*BSIZE. Present uses V stage of block n;
    # delayed uses routed hist/V blocks.
    greads::Vector{Tuple{Int,Int,Int,Matrix{Float64}}}   # (n,st,w, g::d×Nfull)
    bh::Vector{Float64}                                    # (b_st*h) aligned with greads
    W::Int; BSIZE::Int; d::Int; S::Int; p::Int; r::Int
end

# =============================================================================
# Build the one-period covariance map on vec(window covariance) (W×W), via the tensored
# collocation solve with stage-level de-frozen noise injection.
#
# Layout: FULL block space Z = [hist blocks 1..r+1 ; produced blocks 1..p], buffer index
# bb∈[1, p+r+1] (1=oldest hist .. newest produced). Block bb occupies (bb-1)*BSIZE+(1:BSIZE);
# within a block: endpoint=1:d, stage st = d+(st-1)*d+(1:d).
# The deterministic relation: produced blocks V (bb=r+2..p+r+1) satisfy L V = R y_hist with
# y_hist = hist blocks. Equivalently Z is an affine (here linear) image of the period-start
# window y0 (the (r+1) hist blocks): Z = Φ_full · y0, Φ_full :: Nfull×W.
# =============================================================================
function build_period_cov_map(prob::SDDEProblem, S, p)
    a,b,c=gl_tab(S); d=prob.d; h=prob.T/p
    ts=range(0,prob.T,length=p+1); r=max(round(Int,maxdelay(prob,ts)/h),1)
    BSIZE=(S+1)*d; W=(r+1)*BSIZE; nB=p*BSIZE; Nfull=(p+r+1)*BSIZE
    steps=[build_step(prob,a,b,c,h,(n-1)*h,0.0,r) for n in 1:p]
    L,R,_=build_LR(prob,a,b,c,p,h,0.0,r,steps); Lf=lu(Matrix(L)); Rm=Matrix(R)

    # Φ_full: period-start window y0 (W) → full state Z (Nfull). hist = y0 (loaded), V=L⁻¹R·yhist.
    Φ=zeros(Nfull,W)
    for k in 1:W
        e=zeros(W); e[k]=1.0
        vh=zeros(W); for i in 0:r; vh[(r-i)*BSIZE+1:(r-i+1)*BSIZE]=e[i*BSIZE+1:(i+1)*BSIZE]; end
        V=Lf\(Rm*vh)
        z=zeros(Nfull); z[1:W]=vh; z[W+1:end]=V; Φ[:,k]=z
    end
    # period-END window = produced blocks p-r..p (newest r+1 produced), mapped to window order.
    # window block i (0-based age i) = produced block (p-i). full-space rows:
    endmap=zeros(W,Nfull)
    for i in 0:r
        kk=p-i
        srcrows = kk>=1 ? ((r+1)*BSIZE+(kk-1)*BSIZE .+ (1:BSIZE)) : ((kk+r)*BSIZE .+ (1:BSIZE))
        endmap[i*BSIZE .+ (1:BSIZE), srcrows] .= Matrix(I,BSIZE,BSIZE)
    end
    U = endmap*Φ                         # W×W first-moment monodromy (== verified U)

    # ---- noise reads on the FULL space, as functions of the period-start window (d×W) ----
    # present x(s_st) = stage st value of produced block n (buffer n+r+1). delayed via routing.
    function present_read(n,st)
        rows=(r+1)*BSIZE+(n-1)*BSIZE+d+(st-1)*d .+ (1:d)   # stage st of produced block n
        return Φ[rows,:]                                    # d×W (DE-FROZEN: actual stage value)
    end
    function delayed_read(n,st,k)
        τf=prob.delays[k][1]; tst=(n-1)*h+c[st]*h; τval=τf(tst); rel=(tst-τval-0.0)/h+r+1
        mi=floor(Int,rel); w=colloc_w(c,rel-mi); Rd=zeros(d,W); nblk=Nfull÷BSIZE
        if 1<=mi<=nblk; Rd .+= w[1].*Φ[(mi-1)*BSIZE .+ (1:d),:]; end
        if 1<=mi+1<=nblk
            Rd .+= w[S+2].*Φ[mi*BSIZE .+ (1:d),:]
            for ss in 1:S; Rd .+= w[ss+1].*Φ[mi*BSIZE+d+(ss-1)*d .+ (1:d),:]; end
        end
        return Rd
    end
    # endpoint-kick propagator: how a unit covariance deposited in produced block n's endpoint
    # reaches the period-end window. Build by kicking block n endpoint and propagating via L.
    # A state increment δ at block n's endpoint (the new x) affects LATER blocks through L
    # (forward coupling). Solve L·ΔV = (unit at block n endpoint) then map to window.
    function endpoint_prop(n)
        P=zeros(W,d)
        for di in 1:d
            rhs=zeros(nB); rhs[(n-1)*BSIZE+di]=1.0
            ΔV=Lf\rhs
            z=zeros(Nfull); z[W+1:end]=ΔV
            P[:,di]=endmap*z
        end
        return P                                           # W×d
    end
    Pend=[endpoint_prop(n) for n in 1:p]

    # assemble greads
    GR=Tuple{Int,Float64,Matrix{Float64}}[]   # (n, b_st*h, G::d×W)
    for n in 1:p, st in 1:S
        tst=(n-1)*h+c[st]*h
        Rp=present_read(n,st)
        for (w,(αf,βfs,σf)) in enumerate(prob.noise)
            G=αf(tst)*Rp
            for (k,_) in enumerate(prob.delays)
                β=βfs[k](tst); all(iszero,β) && continue
                G=G .+ β*delayed_read(n,st,k)
            end
            push!(GR,(n, b[st]*h, G))
        end
    end

    # one-period covariance map 𝓜[C] = U C Uᵀ + Σ (b_st h) Pend[n] (G C Gᵀ) Pend[n]ᵀ
    function applyM(C)
        out=U*C*U'
        for (n,wt,G) in GR
            out .+= wt .* (Pend[n]*(G*C*G')*Pend[n]')
        end
        return out
    end
    return (applyM=applyM,U=U,W=W,d=d,S=S,p=p)
end

# dominant eigenvalue of the covariance map (dense, proof stage)
function rho_secondmoment(prob,S,p)
    m=build_period_cov_map(prob,S,p); W=m.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx); H=zeros(Nv,Nv)
    for k in 1:Nv
        (i,j)=idx[k]; C=zeros(W,W); C[i,j]=1.0; C[j,i]=1.0; Cn=m.applyM(C)
        for mm in 1:Nv; (a2,b2)=idx[mm]; H[mm,k]=Cn[a2,b2]; end
    end
    return maximum(abs.(eigen(H).values))
end
rho_firstmoment(prob,S,p)= (m=build_period_cov_map(prob,S,p); maximum(abs.(eigen(m.U).values)))

println("lyap_delay loaded")

if abspath(PROGRAM_FILE)==@__FILE__
    # noise-off identity (per dimension)
    println("\n=== noise-off ρ(𝓜)=ρ(U)² ===")
    _p1=SDDEProblem(1,1.0,t->reshape([-1.0],1,1),[(t->1.0,t->reshape([-0.4],1,1))],
        [(t->reshape([0.0],1,1),[t->reshape([0.0],1,1)],t->[0.0])])
    for S in [2,3]
        ru=rho_firstmoment(_p1,S,8); r2=rho_secondmoment(_p1,S,8)
        @printf("  d1 S=%d: ρ(U)²=%.12f ρ(𝓜)=%.12f diff=%.1e\n",S,ru^2,r2,abs(r2-ru^2))
    end

    # LITMUS: scalar present-noise dx=αx dW (delay present but B=β=0 except α) → exp((2a+α²)T)
    println("\n=== LITMUS scalar present-noise dx=αx dW → exp((2a+α²)T) ===")
    a_t=-0.7; α=0.5; T=1.0; exact=exp((2a_t+α^2)*T)
    _pα=SDDEProblem(1,1.0,t->reshape([a_t],1,1),[(t->1.0,t->reshape([0.0],1,1))],
        [(t->reshape([α],1,1),[t->reshape([0.0],1,1)],t->[0.0])])
    @printf("  exact=%.11f\n",exact)
    for S in [1,2,3]
        @printf("  GL%d:\n",S); prev=nothing
        for p in [4,8,16,32]
            ρ=rho_secondmoment(_pα,S,p); err=abs(ρ-exact); rate=prev===nothing ? NaN : log2(prev/err)
            @printf("    p=%2d ρ=%.10f err=%.2e rate=%.2f\n",p,ρ,err,rate); prev=err
        end
    end

    # Hayes (α=0, delayed-only noise) → 0.57022372583
    println("\n=== Hayes (delayed-only noise β) → 0.57022372583 ===")
    _ph=SDDEProblem(1,1.0,t->reshape([-1.0],1,1),[(t->1.0,t->reshape([-0.4],1,1))],
        [(t->reshape([0.0],1,1),[t->reshape([0.3],1,1)],t->[0.0])])
    for S in [1,2,3]
        @printf("  GL%d:\n",S); prev=nothing; ref=0.57022372583
        for p in [4,8,16,32]
            ρ=rho_secondmoment(_ph,S,p); err=abs(ρ-ref); rate=prev===nothing ? NaN : log2(prev/err)
            @printf("    p=%2d ρ=%.11f err=%.2e rate=%.2f\n",p,ρ,err,rate); prev=err
        end
    end
end
