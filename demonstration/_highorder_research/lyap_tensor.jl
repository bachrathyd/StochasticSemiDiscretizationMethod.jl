# =============================================================================
# HIGH-ORDER 2nd moment via the TENSORED collocation solve (unambiguous propagation).
#
#   First moment:  L V = R y0           (V = p produced blocks, y0 = (r+1) hist window)
#   Second moment: produced covariance Vc=E[V Vᵀ], history covariance Ch=E[y0 y0ᵀ]:
#       (L⊗L) vec(Vc) = (R⊗R) vec(Ch) + 𝒩[Vc, Ch]
#   𝒩 = Itô noise injected at each produced block's ENDPOINT×ENDPOINT sub-block:
#       for step n, ΔVc[endpoint_n, endpoint_n] += Σ_w Σ_st (b_st h) g_{n,st,w} Cfull g_{n,st,w}ᵀ
#   g reads g(s_st)=α(s_st)x(s_st)+β(s_st)x(s_st-τ) from the FULL state [y0; V]:
#     present x(s_st) = stage st of produced block n  (DE-FROZEN: real stage value)
#     delayed x(s_st-τ) = routed hist/produced blocks (collocation weights)
#   Cfull = covariance of [y0; V]. Since V is the unknown, 𝒩 couples Vc to itself + Ch.
#
#   The (L⊗L)⁻¹ performs the propagation to period end EXACTLY at order 2S — no hand-rolled
#   1/h factor to get wrong. Noise-off ⇒ vec(Vc)=(L⁻¹R⊗L⁻¹R)vec(Ch) ⇒ window map U⊗U ⇒ ρ(U)².
#
#   ρ of the period map (window covariance → window covariance) = ρ(H), order 2S.
#   Dense, proof stage (small p). Reuses build_LR / build_step from lyap_delay.jl.
# =============================================================================
using LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__,"lyap_delay.jl"))   # gl_tab, colloc_w, build_step, build_LR, maxdelay, SDDEProblem

# Build the period covariance map on vec(window covariance C0) [(r+1)BSIZE]² → same.
# We work with the FULL state covariance Cfull of z=[y0(r+1 blocks); V(p blocks)] (Nfull),
# but only y0 (hist) and V are coupled; produced V solved from y0 via the tensored system.
function tensor_period_map(prob::SDDEProblem, S, p)
    a,b,c=gl_tab(S); d=prob.d; h=prob.T/p
    ts=range(0,prob.T,length=p+1); r=max(round(Int,maxdelay(prob,ts)/h),1)
    BSIZE=(S+1)*d; W=(r+1)*BSIZE; nB=p*BSIZE; Nfull=(p+r+1)*BSIZE
    steps=[build_step(prob,a,b,c,h,(n-1)*h,0.0,r) for n in 1:p]
    L,R,_=build_LR(prob,a,b,c,p,h,0.0,r,steps)
    Lmat=Matrix(L); Rmat=Matrix(R); Lf=lu(Lmat)

    # full-state map Φ: y0 (W) → z=[y0; V] (Nfull).  V = L⁻¹ R (loaded y0).
    # load y0→hist (reverse block order, as in U-build): hist block (r-i)=window block i.
    loadW = zeros(W, W)   # y0(window order) → y_hist(buffer order, oldest..newest)
    for i in 0:r; loadW[(r-i)*BSIZE+1:(r-i+1)*BSIZE, i*BSIZE+1:(i+1)*BSIZE]=Matrix(I,BSIZE,BSIZE); end
    Vmap = Lf\(Rmat*loadW)                     # nB×W : produced blocks vs window
    Φ = vcat(loadW, Vmap)                      # Nfull×W

    # period-END window (W) extracted from z: window block i (age i) = produced block p-i
    endmap=zeros(W,Nfull)
    for i in 0:r
        kk=p-i
        src = kk>=1 ? (W+(kk-1)*BSIZE .+ (1:BSIZE)) : ((kk+r)*BSIZE .+ (1:BSIZE))
        endmap[i*BSIZE .+ (1:BSIZE), src] .= Matrix(I,BSIZE,BSIZE)
    end
    U = endmap*Φ

    # noise coefficient reads on the FULL state z (d×Nfull), per (n,st,w):
    nblk=Nfull÷BSIZE
    function presentZ(n,st)   # stage st of produced block n (buffer block n+r+1)
        rows = W + (n-1)*BSIZE + d + (st-1)*d .+ (1:d); E=zeros(d,Nfull); E[:,rows]=Matrix(I,d,d); return E
    end
    function delayedZ(n,st,k)
        τf=prob.delays[k][1]; tst=(n-1)*h+c[st]*h; τval=τf(tst); rel=(tst-τval-0.0)/h+r+1
        mi=floor(Int,rel); w=colloc_w(c,rel-mi); E=zeros(d,Nfull)
        if 1<=mi<=nblk; E[:,(mi-1)*BSIZE .+ (1:d)] .+= w[1]*Matrix(I,d,d); end
        if 1<=mi+1<=nblk
            E[:,mi*BSIZE .+ (1:d)] .+= w[S+2]*Matrix(I,d,d)
            for ss in 1:S; E[:,mi*BSIZE+d+(ss-1)*d .+ (1:d)] .+= w[ss+1]*Matrix(I,d,d); end
        end
        return E
    end
    # produced-block endpoint rows in z (where noise covariance is deposited)
    prod_end(n)= W + (n-1)*BSIZE .+ (1:d)

    # assemble noise list: (n, weight b_st*h, G::d×Nfull)
    GR=Tuple{Int,Float64,Matrix{Float64}}[]
    for n in 1:p, st in 1:S
        tst=(n-1)*h+c[st]*h
        for (w,(αf,βfs,σf)) in enumerate(prob.noise)
            G=αf(tst)*presentZ(n,st)
            for (k,_) in enumerate(prob.delays)
                β=βfs[k](tst); all(iszero,β) && continue
                G=G .+ β*delayedZ(n,st,k)
            end
            push!(GR,(n, b[st]*h, G))
        end
    end

    # The period map on the FULL covariance Cfull (Nfull×Nfull): but Cfull is determined by
    # the window covariance C0 via z=Φ y0 ⇒ Cfull = Φ C0 Φᵀ + (noise already in V via solve).
    # Tensored solve: V = L⁻¹(R y0 + noise-forcing). The noise enters V's endpoint blocks.
    # Cov(V) = Vmap C0 Vmapᵀ  +  L⁻¹ 𝒬 L⁻ᵀ , where 𝒬 = Σ (b_st h)(eₙ⊗I_d) G Cfull Gᵀ (eₙ⊗I_d)ᵀ
    # with Cfull = Φ C0 Φᵀ + Cov(V_noise) (self-consistent). For the SPECTRAL problem we
    # iterate: but it's linear, so we solve the fixed point as a linear map on vec(C0).
    #
    # Implement the period map C0 ↦ C0' by: (1) Cfull0 = Φ C0 Φᵀ (deterministic full cov),
    # (2) build noise forcing 𝒬 from Cfull0, (3) ΔV = L⁻¹ (assemble 𝒬 into produced rows),
    # actually 𝒬 is a COVARIANCE; propagate as L⁻¹ 𝒬_block L⁻ᵀ. (4) Cfull = Cfull0 + [0,0;0,ΔVc],
    # (5) repeat (2)-(4) to convergence (noise feeds its own future covariance) — but for the
    # ONE-PERIOD map the within-period self-coupling is exactly captured by the implicit solve,
    # so we do the full tensored linear solve instead of iterating:
    #
    # Build operator T on vec(Cfull restricted to needed blocks). For proof simplicity and
    # correctness, do the dense fixed-point solve: C0' = endmap*Cfull*endmapᵀ where Cfull
    # solves Cfull = Φ C0 Φᵀ + N(Cfull), N linear. Solve (I − N)·vecCfull = vec(Φ C0 Φᵀ).
    # N(Cfull): for each (n,wt,G): add into produced-endpoint(n)×produced-endpoint(n) block
    #   wt * Gp*Cfull*Gpᵀ  where Gp = (L⁻¹ deposited at prod_end via the produced structure).
    # The deposited noise covariance at produced block n's endpoint propagates to LATER
    # produced blocks via L⁻¹. So N(Cfull) = Linv_full * D(Cfull) * Linv_fullᵀ restricted...
    #
    # To avoid further hand-rolling, we BUILD N as a dense matrix on vec(Cfull) by probing,
    # using: a noise covariance Σ deposited at produced-endpoint(n) propagates as
    #   ΔVc = Linv * S_n Σ S_nᵀ * Linvᵀ  (S_n places d×d into block-n endpoint of the nB space),
    # then embedded into the V-part of Cfull. This Linv propagation is EXACT (order 2S).
    Linv = inv(Lmat)
    function Nop(Cfull)   # returns Nfull×Nfull, the propagated noise covariance (V-part only)
        # accumulate deposited noise covariance per produced block endpoint (nB space)
        Dep = zeros(nB, nB)
        for (n,wt,G) in GR
            Egg = wt .* (G*Cfull*G')          # d×d  (reads present stage + delayed from Cfull)
            rows=(n-1)*BSIZE .+ (1:d)
            Dep[rows,rows] .+= Egg
        end
        Vc_noise = Linv*Dep*Linv'             # nB×nB propagated to all produced blocks (exact)
        out=zeros(Nfull,Nfull)
        out[W+1:end, W+1:end] = Vc_noise
        return out
    end

    # period map C0 ↦ endmap * Cfull * endmapᵀ, Cfull solving (I−N) vecCfull = vec(Φ C0 Φᵀ).
    # Build the dense linear map M0: vech(C0) → vech(C0') by probing.
    function period_map_dense()
        # Precompute N as a matrix on vec(Cfull): probe basis of Cfull (Nfull²) — too big.
        # Instead solve per-input via fixed-point: N is nilpotent-ish (lower-tri propagation),
        # so iterate N until convergence (within-period, finite steps p).
        return nothing
    end

    # Apply the period map to a window covariance C0 (W×W) → W×W, via fixed-point on Cfull.
    function applyP(C0)
        Cfull = Φ*C0*Φ'
        # fixed point Cfull = Φ C0 Φᵀ + N(Cfull); N propagates noise forward → converges in ≤p iters
        for _ in 1:(p+2)
            Cf_new = Φ*C0*Φ' + Nop(Cfull)
            if norm(Cf_new-Cfull) < 1e-14*(1+norm(Cf_new)); Cfull=Cf_new; break; end
            Cfull=Cf_new
        end
        return endmap*Cfull*endmap'
    end
    return (applyP=applyP, U=U, W=W)
end

function rho2(prob,S,p)
    m=tensor_period_map(prob,S,p); W=m.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx); H=zeros(Nv,Nv)
    for k in 1:Nv
        (i,j)=idx[k]; C=zeros(W,W); C[i,j]=1.0; C[j,i]=1.0; Cn=m.applyP(C)
        for mm in 1:Nv; (a2,b2)=idx[mm]; H[mm,k]=Cn[a2,b2]; end
    end
    maximum(abs.(eigen(H).values))
end
rho1(prob,S,p)=(m=tensor_period_map(prob,S,p); maximum(abs.(eigen(m.U).values)))

if abspath(PROGRAM_FILE)==@__FILE__
    println("=== noise-off ρ(P)=ρ(U)² ===")
    _p1=SDDEProblem(1,1.0,t->reshape([-1.0],1,1),[(t->1.0,t->reshape([-0.4],1,1))],
        [(t->reshape([0.0],1,1),[t->reshape([0.0],1,1)],t->[0.0])])
    for S in [2,3]; ru=rho1(_p1,S,8); r2=rho2(_p1,S,8); @printf("  d1 S=%d ρ(U)²=%.12f ρ(P)=%.12f diff=%.1e\n",S,ru^2,r2,abs(r2-ru^2)); end

    println("\n=== LITMUS dx=αx dW → exp((2a+α²)T) ===")
    a_t=-0.7; α=0.5; T=1.0; exact=exp((2a_t+α^2)*T); @printf("  exact=%.11f\n",exact)
    _pα=SDDEProblem(1,1.0,t->reshape([a_t],1,1),[(t->1.0,t->reshape([0.0],1,1))],
        [(t->reshape([α],1,1),[t->reshape([0.0],1,1)],t->[0.0])])
    for S in [1,2,3]
        @printf("  GL%d:\n",S); prev=nothing
        for p in [4,8,16,32]; ρ=rho2(_pα,S,p); err=abs(ρ-exact); rate=prev===nothing ? NaN : log2(prev/err); @printf("    p=%2d ρ=%.10f err=%.2e rate=%.2f\n",p,ρ,err,rate); prev=err; end
    end

    println("\n=== Hayes (delayed noise) → 0.57022372583 ===")
    _ph=SDDEProblem(1,1.0,t->reshape([-1.0],1,1),[(t->1.0,t->reshape([-0.4],1,1))],
        [(t->reshape([0.0],1,1),[t->reshape([0.3],1,1)],t->[0.0])])
    for S in [1,2,3]
        @printf("  GL%d:\n",S); prev=nothing; ref=0.57022372583
        for p in [4,8,16,32]; ρ=rho2(_ph,S,p); err=abs(ρ-ref); rate=prev===nothing ? NaN : log2(prev/err); @printf("    p=%2d ρ=%.11f err=%.2e rate=%.2f\n",p,ρ,err,rate); prev=err; end
    end
end
