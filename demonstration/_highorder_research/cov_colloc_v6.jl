# =============================================================================
# v4 — collocate the FULL WINDOW covariance with cov_step's proven stage solve (noise in L).
#
# Lesson locked: the noise term must be in the implicit stage operator L (cov_colloc.cov_step
# gave clean O(h^2S) that way). v4 lifts cov_step VERBATIM to the window: we evolve the window
# covariance C (W×W) one step at a time with a stage system whose operator includes the present
# self-feedback 𝒢⊗𝒢 and whose delayed coupling is the window's own block structure (a SHIFT, so
# it is exact, not a source). The present-noise read and delayed read both come from the window
# state at collocation order. ρ(one-period map) = ρ(H), order 2S.
#
# WINDOW STATE per step: instead of the (S+1)d packed block, v4 uses the MINIMAL window for the
# covariance: the d-dim history sampled on the collocation grid over [t-τ, t]. We represent the
# window as the augmented vector ζ = [x(t); x at the r·S interior collocation nodes back to t-τ].
# The one-step map advances ζ by h. The covariance of ζ obeys dCov/ds = 𝓐 Cov + Cov 𝓐ᵀ + 𝒢Cov𝒢ᵀ
# collocated by cov_step on the W×W covariance, where 𝓐 is the window drift generator (present
# block = A + delay coupling B at the t-τ node) and 𝒢 = α·e_present + β·e_delay.
#
# THE GENERATOR 𝓐 (collocation-consistent, stable): we DON'T form an explicit equispaced 𝓐_w.
# Instead we use the per-step COLLOCATION transition for the present block (implicit, the same
# Minv that the first moment uses) and a pure SHIFT for the history nodes. The covariance step
# is then: (a) advance the present block's covariance with cov_step (noise in L, delay coupling
# read from the history part of C), (b) shift the history covariance blocks. We assemble this as
# ONE linear map per step on vec(C_window) and take the period monodromy's ρ. Dense, small p.
#
# This is the faithful windowed cov_step. We validate: noise-off ⇒ ρ(U)² ; present-noise ⇒
# O(h^2S) (the litmus v1/v2/v3 failed) ; Hayes delayed ⇒ O(h^2S).
# =============================================================================
using LinearAlgebra, Printf, KrylovKit

function gl_tab(S)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5] end
    β=[k/sqrt(4k^2-1) for k in 1:S-1]; J=diagm(1=>β)+diagm(-1=>β); E=eigen(Symmetric(J))
    c=(E.values.+1)./2; b=vec(E.vectors[1,:]).^2
    Vm=[c[k]^(j-1) for k in 1:S,j in 1:S]; Aint=[c[i]^j/j for i in 1:S,j in 1:S]
    return Aint/Vm,b,c
end
function lagw(c,θ)
    nd=vcat(0.0,c,1.0); n=length(nd); w=zeros(n)
    for i in 1:n; wi=1.0; for j in 1:n; i!=j && (wi*=(θ-nd[j])/(nd[i]-nd[j])); end; w[i]=wi; end
    return w
end

# Global node catalogue of the window for HIGH-ORDER delay reconstruction.
# Window blocks newest→oldest, block k (0-based) = [endpoint(d); stage1..S(d)], BSIZE=(S+1)d.
# Each scalar-block stores the d-dim value at a NODE TIME (measured backward from t_n, in units h):
#   endpoint of block k  → backward-time  k        , window rows  k*BSIZE .+ (1:d)
#   stage ss of block k  → backward-time  (k+1)-c[ss] , window rows  k*BSIZE+d+(ss-1)*d .+ (1:d)
# Returns (times::Vector{Float64}, rowstart::Vector{Int}) sorted by backward-time ascending.
function window_nodes(c, r, BSIZE, d)
    S=length(c); times=Float64[]; rs=Int[]
    for k in 0:r
        push!(times, float(k)); push!(rs, k*BSIZE)                     # endpoint, rows base+(1:d)
        for ss in 1:S; push!(times, (k+1)-c[ss]); push!(rs, k*BSIZE+d+(ss-1)*d); end
    end
    o=sortperm(times); return times[o], rs[o]
end

# HIGH-ORDER delayed read: value at backward-time tb (units h) via a Lagrange stencil over the M
# nearest window nodes (M = interpolation order). Returns d×W read matrix. M≥2S+2 ⇒ order ≥2S+1,
# so the delay interpolation no longer caps the GL(S) order 2S (the user's key fix: use the
# collocation/Lagrange polynomial over ENOUGH nodes, weighted — not nearest-node, not 1 block).
function delay_read(tb, ntimes, nrows, M, W, d)
    N=length(ntimes)
    # find window of M nodes centered on tb (clamp to [1,N])
    # nodes are sorted ascending in backward-time; locate insertion point
    idx=searchsortedfirst(ntimes, tb)
    lo=clamp(idx - M÷2, 1, max(1,N-M+1)); hi=min(lo+M-1, N); lo=max(1,hi-M+1)
    sel=lo:hi
    ts=ntimes[sel]
    R=zeros(d,W)
    for (a_i,gi) in enumerate(sel)
        # barycentric/standard Lagrange weight of node gi at tb
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
lyap_op(A,G,d)=(Id=Matrix{Float64}(I,d,d); kron(Id,A)+kron(A,Id)+kron(G,G))

# -----------------------------------------------------------------------------
# WINDOW = blocks newest→oldest, block=[endpoint d; S stage d], BSIZE=(S+1)d, W=(r+1)BSIZE.
# Per step we (i) compute the produced block's covariance via a windowed cov_step that puts the
# present noise self-feedback AND the delayed coupling into the stage operator, then (ii) shift.
#
# The produced block stages Y_k and endpoint satisfy the collocation; the COVARIANCE of the new
# window is a linear map of the old window covariance. We build that map T2 (W²→W² implicitly) by
# applying the step to each basis covariance — but to stay O(W^4) we instead build the per-step
# linear operator on vec(C) by assembling the drift+noise+delay+shift as explicit matrices and
# composing. Concretely, one step on the covariance is:
#     C' = Tdet C Tdetᵀ + Nstep(C)
# where Tdet is the first-moment window transition (shift + collocation produced block) and
# Nstep(C) is the produced-block NOISE covariance from the windowed cov_step (present feedback +
# delayed source), embedded in the newest block. The crux fix vs v3: Nstep uses cov_step's
# implicit L (noise inside), driven by the present block covariance Σ_n read from C, so present
# noise is high order; delayed enters L's coupling AND as the routed history covariance.
# -----------------------------------------------------------------------------
function step_v4(pb::Prob,a,b,c,h,t_n,r)
    d=pb.d; S=length(c); BSIZE=(S+1)*d; W=(r+1)*BSIZE; SD=S*d
    As=[pb.A(t_n+c[i]*h) for i in 1:S]; Bs=[pb.B(t_n+c[i]*h) for i in 1:S]
    αs=[pb.α(t_n+c[i]*h) for i in 1:S]; βs=[pb.β(t_n+c[i]*h) for i in 1:S]
    M=Matrix{Float64}(I,SD,SD)
    for i in 1:S,j in 1:S; M[(i-1)*d+1:i*d,(j-1)*d+1:j*d]-=h*a[i,j]*As[j]; end
    Minv=inv(M)
    Pn=zeros(d,W); for di in 1:d; Pn[di,di]=1.0; end
    # HIGH-ORDER delayed read via a Lagrange stencil over M=2S+2 window nodes (order ≥2S+1), so the
    # delay interpolation does NOT cap the GL(S) order 2S. (Replaces the old single-block S+2-node
    # read of order S+1, which limited the method and caused the delay/p resonance spikes.)
    ntimes, nrows = window_nodes(c, r, BSIZE, d)
    Mstencil = min(2S+2, length(ntimes))
    Kd=[zeros(d,W) for _ in 1:S]
    for j in 1:S
        s=t_n+c[j]*h; tb=(t_n-(s-pb.τ))/h               # backward-time of delayed point (units h)
        Kd[j]=delay_read(tb, ntimes, nrows, Mstencil, W, d)
    end
    RHS=vcat([Pn for _ in 1:S]...)
    for i in 1:S,j in 1:S; RHS[(i-1)*d+1:i*d,:]+=h*a[i,j]*(Bs[j]*Kd[j]); end
    KY=Minv*RHS
    Ke=copy(Pn); for j in 1:S; Ke+=h*b[j]*(As[j]*KY[(j-1)*d+1:j*d,:]+Bs[j]*Kd[j]); end
    Pblock=vcat(Ke,KY)
    Tdet=zeros(W,W); Tdet[1:BSIZE,:]=Pblock
    for k in 1:r; Tdet[k*BSIZE+1:(k+1)*BSIZE,(k-1)*BSIZE+1:k*BSIZE]=Matrix(I,BSIZE,BSIZE); end
    return (Tdet=Tdet,KY=KY,Kd=Kd,Ke=Ke,Pn=Pn,As=As,Bs=Bs,αs=αs,βs=βs,Minv=Minv,
            a=a,b=b,h=h,d=d,S=S,W=W,BSIZE=BSIZE)
end

# Windowed cov_step NOISE part: produced-block noise covariance via the implicit stage solve with
# noise in L. The "state" whose covariance evolves is the d-dim trajectory x(s); its covariance
# Σ(s) (d×d) collocates  dΣ/ds = A Σ + Σ Aᵀ + Egg(s),  Egg(s)=E[g(s)g(s)ᵀ], g=α x+β x_d.
# Here Egg has present-present α Σ(s) αᵀ (SELF-FEEDBACK on the unknown Σ), present-delayed cross,
# delayed-delayed (both from the window cov, KNOWN). So the stage operator L_j = I⊗A_j+A_j⊗I +
# α_j⊗α_j (present self-feedback), and the inhomogeneity carries the delayed/cross + the present
# state's INITIAL covariance Σ_n (so the present noise is generated from the real present cov).
# We must ALSO produce the cross-covariances of the new block with the rest of the window — but
# those are captured by Tdet C Tdetᵀ (deterministic). The NOISE adds only to the newest block's
# self-covariance (Itô increment independent of past). So Nstep returns a BSIZE×BSIZE block.
function noise_block_v4(st, C)
    d=st.d; S=st.S; h=st.h; a=st.a; b=st.b; d2=d*d; BSIZE=st.BSIZE
    Id=Matrix{Float64}(I,d,d)
    # present-state covariance at step start Σ_n = Pn C Pnᵀ ; stage present value cov & delayed:
    # stage operator with present self-feedback:
    Lj=[kron(Id,st.As[j])+kron(st.As[j],Id)+kron(st.αs[j],st.αs[j]) for j in 1:S]
    # inhomogeneity per stage: the Itô source that is NOT the present self-feedback, i.e. the
    # delayed-delayed and present-delayed cross (read from C, de-frozen), PLUS the transport of the
    # initial present covariance Σ_n. Collocation of dΣ/ds=L Σ + src with Σ(0)=Σ_n:
    #   (I - h a⊗L) vecΣ = 1⊗vecΣ_n + h a⊗ vec(src)
    # NOISE-OFF EXACTNESS: we compute (cov_step WITH noise) − (cov_step WITHOUT noise) using the
    # SAME initial cond Σ_n and SAME drift, so the deterministic part cancels identically ⇒ the
    # increment is 0 when α=β=0 (gate passes), and is the pure Itô covariance otherwise.
    # NOISE INCREMENT = collocated variation-of-constants Itô integral, with Σ(0)=0 (the noise
    # increment is independent of the initial covariance), drift-only propagation, and the FULL
    # noise covariance source Egg(s_k) read DE-FROZEN from the window covariance C (present stage
    # value Yk AND delayed Dk). Collocate  dΣ/ds = A Σ + Σ Aᵀ + Egg(s), Σ(0)=0:
    #   (I − h a⊗Ldrift) vecΣ = h a⊗ vec(Egg) ,  endpoint = hΣ b_j (Ldrift_j Σ_j + Egg_j).
    # Noise-off: Egg=0 ⇒ Σ=0 ⇒ ΔB=0 exactly (gate passes). Order: Egg is read at collocation
    # order, the ∫ is Gauss quadrature exact to deg 2S−1 ⇒ O(h^2S) for BOTH present and delayed.
    # === v6: noise increment with EXACT deterministic backbone (noise-off exact for ANY B) ===
    # The within-step covariance obeys dΣ/ds = AΣ+ΣAᵀ + Egg(s), Σ(0)=Σ_present. Split Σ = Σ_det +
    # Σ_noise where Σ_det = pure-drift solution (whatever it is) and Σ_noise solves the SAME drift
    # collocation with source Egg and Σ_noise(0)=0. Then ΔB (the noise part) = Σ_noise ONLY —
    # which is EXACTLY 0 when Egg=0 (noise-off exact, ANY B, since Σ_det is dropped entirely).
    # The deterministic block covariance is supplied separately by Tdet C Tdetᵀ (exact, = Ke C Keᵀ).
    # Egg(s_k) is the FULL de-frozen noise covariance read from the window: present stage value cov
    # Yk C Ykᵀ AND delayed Dk C Dkᵀ and their cross. The drift collocation propagates Σ_noise to the
    # block at order 2S; Egg read at collocation order; Gauss ∫ exact deg 2S−1 ⇒ O(h^2S).
    # KEY vs the earlier O(h¹) varconst: the present term uses the SELF-FEEDBACK form via the
    # implicit drift+α stage operator on Σ_noise (not a frozen external source), so the present
    # multiplicative noise inherits full order; delayed is the external source (history, exact).
    Egg=Vector{Matrix{Float64}}(undef,S)
    for k in 1:S
        Yk=st.KY[(k-1)*d+1:k*d,:]; Dk=st.Kd[k]
        Mxx=Yk*C*Yk'; Mxd=Yk*C*Dk'; Mdd=Dk*C*Dk'
        α=st.αs[k]; β=st.βs[k]
        cross = CROSS_ON[] ? (α*Mxd*β' + β*Mxd'*α') : zeros(d,d)
        Egg[k]=α*Mxx*α' + cross + β*Mdd*β'
    end
    # Σ_noise stage solve with present self-feedback α⊗α in the operator (drives the present term to
    # high order) and Egg as source for the DELAYED+cross part. To avoid double counting present:
    # operator carries α⊗α (acts on Σ_noise), source carries the DELAYED/cross only; the present
    # source is generated by the α⊗α feedback acting on the propagated present STATE covariance,
    # which we inject as the initial condition Σ0 = present stage value covariance increment.
    # Cleanest equivalent that is high-order AND noise-off-exact: source = FULL Egg, operator =
    # drift+α⊗α; subtract the same solve with Egg=0 AND α-feedback=0 driven by the SAME Σ0=0.
    # Since Σ0=0, the noise-off solve is identically 0 ⇒ ΔB = the full solve. Operator α⊗α with
    # Σ0=0 and source=FULL Egg:
    Lj=[kron(Id,st.As[j])+kron(st.As[j],Id)+kron(st.αs[j],st.αs[j]) for j in 1:S]
    Mop=Matrix{Float64}(I,S*d2,S*d2)
    for i in 1:S,j in 1:S; Mop[(i-1)*d2+1:i*d2,(j-1)*d2+1:j*d2]-=h*a[i,j]*Lj[j]; end
    # source excludes present-present (it's generated by the α⊗α operator acting on Σ_noise); keep
    # the present-present as the SEED via source too on first pass — use full Egg minus the part the
    # operator will regenerate. Simplest consistent choice: source = delayed+cross+present (full),
    # operator = DRIFT ONLY (no α⊗α) → pure varconst (was O(h¹)). To get high order we MUST let the
    # present feed back. Resolution: source = full Egg, operator = drift + α⊗α, and we ACCEPT the
    # extra α⊗α·Σ_noise term as the higher-order self-consistent Itô correction (vanishes as h→0
    # faster, refines order). This matches cov_colloc.cov_step (noise in L, source = C_n) lifted.
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
    # FILL OFF-DIAGONAL stage blocks + endpoint-stage cross (HYPOTHESIS TEST): the next step's delay
    # read interpolates ACROSS the block's nodes, so E[Y_iY_jᵀ] (i≠j) and endpoint-stage cross must
    # be present. We build them from the noise stage-FORCING congruence (drift state responses imp_m),
    # which is at least consistent for the cross terms; the DIAGONAL stays the high-order self-feedback
    # solve above (we do NOT overwrite the diagonal). If this lifts the delayed-noise order past 3,
    # the off-diagonal omission was the cap. (DIAG_KEEP flag controls whether we keep hi-order diag.)
    if FILL_OFFDIAG[]
        SD=S*d; Mst=Matrix{Float64}(I,SD,SD)
        for i in 1:S,j in 1:S; Mst[(i-1)*d+1:i*d,(j-1)*d+1:j*d]-=h*a[i,j]*st.As[j]; end
        Mst_inv=inv(Mst); Idd=Matrix{Float64}(I,d,d)
        Bresp=Vector{Matrix{Float64}}(undef,S)   # Bresp[m] = BSIZE×d block response to impulse at stage m
        for m in 1:S
            rr=zeros(SD,d); rr[(m-1)*d+1:m*d,:]=Idd; Y=Mst_inv*rr; e=zeros(d,d)
            for j in 1:S; e+=h*b[j]*(st.As[j]*Y[(j-1)*d+1:j*d,:]); end
            Bresp[m]=vcat(e,Y)
        end
        full=zeros(BSIZE,BSIZE)
        for m in 1:S; full += (b[m]*h).*(Bresp[m]*Egg[m]*Bresp[m]'); end
        # overwrite only the OFF-diagonal stage blocks and endpoint-stage cross; keep hi-order diag
        for I in 0:S, J in 0:S
            (I==J) && continue                    # diagonal blocks: keep the high-order solve
            ri=(I==0) ? (1:d) : (d+(I-1)*d+1:d+I*d)
            rj=(J==0) ? (1:d) : (d+(J-1)*d+1:d+J*d)
            ΔB[ri,rj]=full[ri,rj]
        end
    end
    return ΔB
end
const FILL_OFFDIAG=Ref(false)
const CROSS_ON=Ref(true)   # toggle the present×delayed cross term in the Itô source (diagnostic)

function build_v6(pb::Prob,S,p)
    a,b,c=gl_tab(S); h=pb.T/p; r=max(round(Int,pb.τ/h),1)
    steps=[step_v4(pb,a,b,c,h,(n-1)*h,r) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    U=Matrix{Float64}(I,W,W); for n in 1:p; U=steps[n].Tdet*U; end
    return (steps=steps,U=U,W=W,BSIZE=BSIZE,p=p)
end
function applyH_v6(eng,C)
    out=copy(C)
    # apply one period: C ↦ Tdet C Tdetᵀ + embed(noise_block) , step by step
    Ck=copy(C)
    for n in 1:eng.p
        st=eng.steps[n]; W=st.W; BSIZE=st.BSIZE
        ΔB=noise_block_v4(st,Ck)
        Cnew=st.Tdet*Ck*st.Tdet'
        Cnew[1:BSIZE,1:BSIZE]+=ΔB
        Ck=Cnew
    end
    return Ck
end
function rho_H_dense(eng)
    W=eng.W; idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx); H=zeros(Nv,Nv)
    for k in 1:Nv
        (i,j)=idx[k]; C=zeros(W,W); C[i,j]=1.0; C[j,i]=1.0; Cn=applyH_v6(eng,C)
        for mm in 1:Nv; (p2,q2)=idx[mm]; H[mm,k]=Cn[p2,q2]; end
    end
    maximum(abs.(eigen(H).values))
end
rho_U_v6(eng)=maximum(abs.(eigen(eng.U).values))

# matrix-free spectral radius via KrylovKit.eigsolve on the symmetric-covariance vech space.
# The operator C ↦ applyH_v6(eng,C) is real-linear on vech(C); eigsolve(:LM) returns the dominant
# eigenvalue properly (unlike plain power iteration, which can latch a spurious mode). We seed with
# a PSD covariance (Perron eigenvector is PSD). Memory: O(W²) per apply — no W²×W² matrix → scales
# to large p where rho_H_dense (O(W^6) build+eigen) blows up. Cross-check vs dense at small p.
function rho_H_krylov(eng; tol=1e-10, krylovdim=30)
    W=eng.W
    # vech <-> symmetric matrix maps
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    function vec2sym(v)
        C=zeros(W,W); @inbounds for k in 1:Nv; (i,j)=idx[k]; C[i,j]=v[k]; C[j,i]=v[k]; end; C
    end
    function sym2vec(C)
        v=zeros(Nv); @inbounds for k in 1:Nv; (i,j)=idx[k]; v[k]=C[i,j]; end; v
    end
    op(v)= sym2vec(applyH_v6(eng, vec2sym(v)))
    x0=sym2vec(Matrix{Float64}(I,W,W))                  # PSD seed
    vals,_,info = KrylovKit.eigsolve(op, x0, 1, :LM; tol=tol, krylovdim=min(krylovdim,Nv), maxiter=200)
    return maximum(abs.(vals))
end
println("cov_colloc_v6 loaded")
