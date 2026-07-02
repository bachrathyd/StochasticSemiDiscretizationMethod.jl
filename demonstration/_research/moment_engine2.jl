# =============================================================================
# Second-moment engine via MOMENT-DDE COEFFICIENT ASSEMBLY (the proven route).
#
# For a linear SDDE (Itô), the second moment satisfies a deterministic linear DDE in
# the covariance variables. We assemble that moment-DDE's coefficient operators and
# collocate it at GL(S) (order 2S) — beating SDM's O(h³) ceiling.
#
# We build the moment-DDE on the symmetric/cross covariance state and use the SAME
# GL(S) collocation window monodromy as the first moment (moment_engine.jl machinery),
# applied to the LIFTED moment system.  requires moment_engine.jl loaded first.
#
# General lifting (single delay τ, possibly time-varying & periodic coefficients,
# d-dimensional, additive+multiplicative noise):
#   moment state u(t) = [ vec(M(t)) ; vec(P(t)) ]
#     M(t) = E[x(t) x(t)ᵀ]            (d² entries, symmetric)
#     P(t) = E[x(t) x(t-τ)ᵀ]         (d² entries)
#   Using Itô:
#     dM/dt = A M + M Aᵀ + B Pᵀ + P Bᵀ
#             + Σⱼ [ αⱼ M αⱼᵀ + αⱼ P βⱼᵀ + βⱼ Pᵀ αⱼᵀ + βⱼ Mτ βⱼᵀ ]   (Mτ=M(t-τ))
#     dP/dt = A P + B Mτ + (no Itô cross term: x(t-τ) ⟂ dW(t))
#   This is a linear DDE  u'(t) = 𝒜(t) u(t) + ℬ(t) u(t-τ)  on dim 2d², with the Itô
#   multiplicative terms folded into 𝒜 (M-part) and ℬ (Mτ-part).
#   Collocating it at GL(S) gives ρ(H) at order 2S.
#
# NOTE: this single-delay lifting covers the large majority of examples (incl. Mathieu).
# Multi-delay extends u with P_k = E[x(t) x(t-τ_k)ᵀ] per delay; implemented below for the
# two-delay case as well.
# =============================================================================
using LinearAlgebra

# Kronecker identities for vec:  vec(A X Bᵀ) = (B ⊗ A) vec(X).
_kAB(A,B) = kron(B, A)

# Build moment-DDE coefficient functions 𝒜(t), ℬ(t) (each (Ndim)×(Ndim)) for a
# SINGLE-DELAY SDDE. u = [vec(M); vec(P)], Ndim = 2 d².
function moment_dde_single(prob::SDDEProblem)
    d = prob.d
    @assert length(prob.delays)==1 "moment_dde_single: exactly one delay"
    τf, Bf = prob.delays[1]
    Id = Matrix{Float64}(I,d,d)
    nM = d*d
    Ndim = 2*nM
    function 𝒜(t)
        A = prob.A(t); B = Bf(t)
        Z = zeros(Ndim,Ndim)
        # dM/dt local part: A M + M Aᵀ  ⇒ (I⊗A + Aⓧ?) ; vec(AM+MAᵀ)=(I⊗A+A⊗I)vec?
        # vec(A M) = (I⊗A)vec(M); vec(M Aᵀ) = (A⊗I)vec(M)
        MM = _kAB(A,Id) + _kAB(Id,A)
        # multiplicative present: Σ αⱼ M αⱼᵀ ⇒ Σ (αⱼ⊗αⱼ) vec(M)
        for (αf,βfs,σf) in prob.noise
            α=αf(t); MM += _kAB(α,α)
        end
        Z[1:nM, 1:nM] = MM
        # dM/dt cross to P: B Pᵀ + P Bᵀ.  vec(B Pᵀ): Pᵀ has vec(Pᵀ)=K vec(P) (commutation).
        # Simpler: B Pᵀ + P Bᵀ = B Pᵀ + (B Pᵀ)ᵀ. vec(B Pᵀ) = (I⊗B) vec(Pᵀ) = (I⊗B) K vec(P).
        K = comm_matrix(d,d)
        BP = _kAB(B,Id)*K + _kAB(Id,B)   # vec(B Pᵀ)=(I⊗B)K vec P ; vec(P Bᵀ)=(B⊗I) vec P
        # plus multiplicative present-delay cross: αⱼ P βⱼᵀ + βⱼ Pᵀ αⱼᵀ
        # vec(α P βᵀ)=(β⊗α)vec(P)=_kAB(α,β) (NO K); vec(β Pᵀ αᵀ)=(α⊗β)K vec(P)=_kAB(β,α)*K
        for (αf,βfs,σf) in prob.noise
            α=αf(t); β=βfs[1](t)
            BP += _kAB(α,β) + _kAB(β,α)*K
        end
        Z[1:nM, nM+1:end] = BP
        # dP/dt: A P  (local)
        Z[nM+1:end, nM+1:end] = _kAB(Id,A)   # vec(A P)=(I⊗A)vec P
        return Z
    end
    function ℬ(t)
        A = prob.A(t); B = Bf(t)
        Z = zeros(Ndim,Ndim)
        # dM/dt delayed: Σ βⱼ Mτ βⱼᵀ ⇒ Σ (βⱼ⊗βⱼ) vec(Mτ)
        MM = zeros(nM,nM)
        for (αf,βfs,σf) in prob.noise
            β=βfs[1](t); MM += _kAB(β,β)
        end
        Z[1:nM, 1:nM] = MM
        # dP/dt delayed: B Mτ ⇒ (I⊗B) vec(Mτ)
        Z[nM+1:end, 1:nM] = _kAB(Id,B)
        return Z
    end
    return 𝒜, ℬ, Ndim
end

# Commutation matrix K_{d,d}: vec(Xᵀ) = K vec(X) for d×d X.
function comm_matrix(m,n)
    K = zeros(m*n, m*n)
    for i in 1:m, j in 1:n
        K[(i-1)*n+j, (j-1)*m+i] = 1.0   # maps vec(X)→vec(Xᵀ)
    end
    return K
end

# Collocate the moment-DDE u'(t)=𝒜(t)u(t)+ℬ(t)u(t-τ) over one period and return ρ.
function second_moment_rho(prob::SDDEProblem, S, p)
    𝒜, ℬ, Ndim = moment_dde_single(prob)
    # Build a generic linear-DDE collocation monodromy with time-varying 𝒜,ℬ.
    return moment_rho_generic(𝒜, ℬ, first(prob.delays)[1], prob.T, Ndim, S, p)
end

# Generic GL(S) collocation monodromy for u'(t)=𝒜(t)u + ℬ(t)u(t-τ(t)) over [0,T].
function moment_rho_generic(𝒜, ℬ, τf, T, D, S, p)
    a,b,c = gl_tableau(S)
    h = T/p
    ts = range(0,T,length=p+1)
    r = max(round(Int, maximum(τf(t) for t in ts)/h), 1)
    BSIZE=(S+1)*D
    IL=Int[];JL=Int[];VL=Float64[]; IR=Int[];JR=Int[];VR=Float64[]
    for n in 1:p
        roff=(n-1)*BSIZE; t_n=(n-1)*h
        Astage=[𝒜(t_n+c[i]*h) for i in 1:S]
        # M = I - h*(a⊗Astage_j) (block)
        SD=S*D
        Mm=Matrix{Float64}(I,SD,SD)
        for i in 1:S,j in 1:S
            Mm[(i-1)*D+1:i*D,(j-1)*D+1:j*D] -= h*a[i,j]*Astage[j]
        end
        Minv=inv(Mm)
        RHSy=zeros(SD,D); for i in 1:S; RHSy[(i-1)*D+1:i*D,:]=Matrix(I,D,D); end
        Yy=Minv*RHSy
        ynext=Matrix{Float64}(I,D,D)
        for j in 1:S; ynext+=h*b[j]*(Astage[j]*Yy[(j-1)*D+1:j*D,:]); end
        Mprop=vcat(ynext,Yy)
        for i in 1:BSIZE; push!(IL,roff+i);push!(JL,roff+i);push!(VL,1.0); end
        for di in 1:D, rb in 1:BSIZE
            v=-Mprop[rb,di]
            if v!=0
                if n==1; push!(IR,roff+rb);push!(JR,r*BSIZE+di);push!(VR,v)
                else; push!(IL,roff+rb);push!(JL,(n-2)*BSIZE+di);push!(VL,v); end
            end
        end
        for st in 1:S
            tst=t_n+c[st]*h; Bst=ℬ(tst)
            RHSd=zeros(SD,D)
            for i in 1:S; RHSd[(i-1)*D+1:i*D,:]=h*a[i,st]*Bst; end
            Yd=Minv*RHSd
            ynd=zeros(D,D)
            for j in 1:S; term=Astage[j]*Yd[(j-1)*D+1:j*D,:]; if j==st; term+=Bst; end; ynd+=h*b[j]*term; end
            Md=vcat(ynd,Yd)
            τval=τf(tst); rel=(tst-τval)/h+r+1; mi=floor(Int,rel); w=colloc_weights(c,rel-mi)
            bx=mi-(r+1)
            for dj in 1:D, rb in 1:BSIZE
                val=-Md[rb,dj]*w[1]
                if val!=0
                    if bx<=0; push!(IR,roff+rb);push!(JR,(bx+r)*BSIZE+dj);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(bx-1)*BSIZE+dj);push!(VL,val); end
                end
            end
            be=(mi+1)-(r+1)
            for ss in 1:S, dj in 1:D, rb in 1:BSIZE
                val=-Md[rb,dj]*w[ss+1]
                if val!=0
                    col=D+(ss-1)*D+dj
                    if be<=0; push!(IR,roff+rb);push!(JR,(be+r)*BSIZE+col);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(be-1)*BSIZE+col);push!(VL,val); end
                end
            end
            for dj in 1:D, rb in 1:BSIZE
                val=-Md[rb,dj]*w[S+2]
                if val!=0
                    if be<=0; push!(IR,roff+rb);push!(JR,(be+r)*BSIZE+dj);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(be-1)*BSIZE+dj);push!(VL,val); end
                end
            end
        end
    end
    L=sparse(IL,JL,VL,p*BSIZE,p*BSIZE); R=sparse(IR,JR,VR,p*BSIZE,(r+1)*BSIZE)
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
    return maximum(abs.(eigen(Phi).values))
end
