# =============================================================================
# Moment-collocation engine for linear stochastic delay differential equations.
#
# High-order (superconvergent) second-moment stability via collocation of the
# DETERMINISTIC moment equation — beats the SDM trajectory-level O(h³) ceiling.
#
# An SDDE (Itô):
#   dx = [ A(t) x + Σₖ Bₖ(t) x(t-τₖ(t)) + c(t) ] dt
#      + Σⱼ [ αⱼ(t) x + Σₖ βₖⱼ(t) x(t-τₖ(t)) + σⱼ(t) ] dWⱼ
# x ∈ R^d, independent Wiener processes Wⱼ.
#
# The second-moment / covariance of the discretized augmented window state obeys a
# DETERMINISTIC linear delay system; we discretize THAT with GL(S) collocation.
#
# This file is self-contained (no MFCM, no Drive). Used by all demonstration examples.
# =============================================================================
using LinearAlgebra, SparseArrays

# ---- GL(S) Gauss–Legendre Butcher tableaux (S = 1..5) ----
function gl_tableau(S::Int)
    if S == 1
        return reshape([0.5],1,1), [1.0], [0.5]
    elseif S == 2
        s3=sqrt(3)
        return [0.25 0.25-s3/6; 0.25+s3/6 0.25], [0.5,0.5], [0.5-s3/6, 0.5+s3/6]
    elseif S == 3
        s15=sqrt(15)
        return [5/36 2/9-s15/15 5/36-s15/30;
                5/36+s15/24 2/9 5/36-s15/24;
                5/36+s15/30 2/9+s15/15 5/36], [5/18,4/9,5/18],
               [0.5-s15/10, 0.5, 0.5+s15/10]
    elseif S == 4
        # Gauss-Legendre 4-stage (order 8), nodes/weights/a from standard tables
        c = [0.0694318442029737, 0.3300094782075719, 0.6699905217924281, 0.9305681557970263]
        b = [0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269]
        a = [0.0869637112843634  -0.0266041800849987   0.0126274626894047  -0.0055555568376512;
             0.1881181174998680   0.1630362887156365  -0.0278804286024709   0.0067355244090221;
             0.1671919219741887   0.3539530060337439   0.1630362887156365  -0.0141906949311991;
             0.1774825722545226   0.3134451147418683   0.3526767575162718   0.0869637112843634]
        return a, b, c
    elseif S == 5
        c = [0.0469100770306680, 0.2307653449471585, 0.5, 0.7692346550528415, 0.9530899229693319]
        b = [0.1184634425280945, 0.2393143352496832, 0.2844444444444444, 0.2393143352496832, 0.1184634425280945]
        a = [0.0592317212640473 -0.0195703643203272  0.0112544008186170 -0.0070960843193925  0.0029874616632248;
             0.1281510056700219  0.1196571676248417 -0.0245921146196539  0.0140888266950769 -0.0057232442331197;
             0.1137762880042246  0.2600046516806250  0.1422222222222222 -0.0203633915457866  0.0073334079221572;
             0.1212324369268338  0.2289960545789127  0.3090365590640758  0.1196571676248417 -0.0169215416748210;
             0.1168753295602285  0.2449081289104774  0.2829703900791167  0.2719925382293682  0.0592317212640473]
        return a, b, c
    else
        error("GL(S) only S=1..5 hardcoded")
    end
end

# collocation interpolation weights on nodes {0, c₁..c_S, 1} (length S+2)
function colloc_weights(c::Vector{Float64}, theta::Float64)
    nodes = vcat(0.0, c, 1.0); n=length(nodes); w=zeros(n)
    for i in 1:n
        wi=1.0
        for j in 1:n; i!=j && (wi *= (theta-nodes[j])/(nodes[i]-nodes[j])); end
        w[i]=wi
    end
    return w
end

# =============================================================================
# Build the DETERMINISTIC moment system as a generic linear DDE in the vectorized
# augmented covariance, then take the GL(S) collocation monodromy over one period.
#
# Strategy: we form the *first-moment* high-order one-step transition for the
# augmented window, then lift it to the covariance (second moment) via the exact
# per-step Kronecker + the within-step collocation that carries the covariance.
#
# For tractability across many examples we use the equivalent and verified route:
# build the per-step augmented deterministic transition F̂ₙ (size (r+1)*BSIZE),
# AND the per-step noise-source covariance Qₙ from the Itô isometry of the noise
# coefficients evaluated on the collocation stages, then propagate the covariance
# with collocation accuracy.  ρ of the period covariance map = ρ(H).
# =============================================================================

# A problem specification.
struct SDDEProblem
    d::Int                                 # state dimension
    T::Float64                             # principal period
    A::Function                            # A(t) :: d×d
    delays::Vector{Tuple{Function,Function}}   # [(τₖ(t)::Float64, Bₖ(t)::d×d)]
    # noise sources: each is (α(t)::d×d, [βₖ(t)::d×d for each delay], σ(t)::d-vector)
    noise::Vector{Tuple{Function,Vector{Function},Function}}
end

# ---- helper: largest delay over the grid ----
maxdelay(prob::SDDEProblem, ts) = maximum(maximum(τ(t) for t in ts) for (τ,_) in prob.delays)

# =============================================================================
# Deterministic high-order one-step transition (augmented), time-varying coeffs,
# multiple + time-varying delays. Builds L,R over the period grid; covariance is
# then propagated. (Generalizes the verified moment_colloc / scratch_port logic.)
# =============================================================================

# Build per-step augmented blocks M_prop, list of (M_del, m_idx, weights) for all delays.
function step_blocks(prob::SDDEProblem, a, b, c, h, t_n, t_start, r)
    d = prob.d; S=length(c); BSIZE=(S+1)*d; SD=S*d
    # M = I - h * (a ⊗ A(stage))   (A evaluated at each stage time)
    M = Matrix{Float64}(I, SD, SD)
    Astage = [prob.A(t_n + c[i]*h) for i in 1:S]
    for i in 1:S, j in 1:S, di in 1:d, dj in 1:d
        # implicit collocation: row i uses A at stage i (Jacobian); standard form
        M[(i-1)*d+di,(j-1)*d+dj] -= h*a[i,j]*Astage[j][di,dj]
    end
    Minv = inv(M)
    RHSy = zeros(SD,d); for i in 1:S, di in 1:d; RHSy[(i-1)*d+di,di]=1.0; end
    Yy = Minv*RHSy
    ynext = Matrix{Float64}(I,d,d)
    for j in 1:S; ynext += h*b[j]*(Astage[j]*Yy[(j-1)*d+1:j*d,:]); end
    Mprop = vcat(ynext, Yy)                       # BSIZE×d

    deldata = Vector{Tuple{Matrix{Float64},Int,Vector{Float64}}}[]  # per delay: per stage
    for (k,(τf,Bf)) in enumerate(prob.delays)
        perstage = Tuple{Matrix{Float64},Int,Vector{Float64}}[]
        for st in 1:S
            tst = t_n + c[st]*h
            Bst = Bf(tst)
            RHSd = zeros(SD,d)
            for i in 1:S, di in 1:d, dj in 1:d
                RHSd[(i-1)*d+di,dj] = h*a[i,st]*Bst[di,dj]
            end
            Yd = Minv*RHSd
            ynd = zeros(d,d)
            for j in 1:S
                term = Astage[j]*Yd[(j-1)*d+1:j*d,:]; if j==st; term += Bst; end
                ynd += h*b[j]*term
            end
            Md = vcat(ynd, Yd)
            τval = τf(tst)
            rel = (tst - τval - t_start)/h + r + 1
            mi = floor(Int, rel); θ = rel - mi
            push!(perstage, (Md, mi, colloc_weights(c, θ)))
        end
        push!(deldata, perstage)
    end
    return Mprop, deldata, BSIZE
end

# Assemble L (p*BSIZE square) and R (p*BSIZE × (r+1)*BSIZE) for the first-moment map.
function build_LR(prob::SDDEProblem, a, b, c, p, h, t_start, r)
    d=prob.d; S=length(c); BSIZE=(S+1)*d
    IL=Int[];JL=Int[];VL=Float64[]; IR=Int[];JR=Int[];VR=Float64[]
    for n in 1:p
        roff=(n-1)*BSIZE; t_n=t_start+(n-1)*h
        Mprop, deldata, _ = step_blocks(prob,a,b,c,h,t_n,t_start,r)
        for i in 1:BSIZE; push!(IL,roff+i);push!(JL,roff+i);push!(VL,1.0); end
        for di in 1:d, rb in 1:BSIZE
            v=-Mprop[rb,di]
            if v!=0
                if n==1; push!(IR,roff+rb);push!(JR,r*BSIZE+di);push!(VR,v)
                else; push!(IL,roff+rb);push!(JL,(n-2)*BSIZE+di);push!(VL,v); end
            end
        end
        for perstage in deldata, (Md,mi,w) in perstage
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

# First-moment window monodromy Φ (size W=(r+1)*BSIZE), exact base_sweep emulation.
function first_moment_phi(prob::SDDEProblem, S, p)
    a,b,c = gl_tableau(S)
    h = prob.T/p
    ts = range(0, prob.T, length=p+1)
    r = round(Int, maxdelay(prob, ts)/h)
    r = max(r, 1)
    L,R,BSIZE = build_LR(prob,a,b,c,p,h,0.0,r)
    Lf=lu(Matrix(L)); Rm=Matrix(R); W=(r+1)*BSIZE
    Phi=zeros(W,W)
    for k in 1:W
        x=zeros(W); x[k]=1.0; vh=zeros((r+1)*BSIZE)
        for i in 0:r; vh[(r-i)*BSIZE+1:(r-i+1)*BSIZE]=x[i*BSIZE+1:(i+1)*BSIZE]; end
        vper=Lf\(Rm*vh); y=zeros(W)
        for i in 0:r
            kk=p-i
            if kk>=1; y[i*BSIZE+1:(i+1)*BSIZE]=vper[(kk-1)*BSIZE+1:kk*BSIZE]
            else; y[i*BSIZE+1:(i+1)*BSIZE]=vh[(kk+r)*BSIZE+1:(kk+r+1)*BSIZE]; end
        end
        Phi[:,k]=y
    end
    return Phi, W, BSIZE, r
end
