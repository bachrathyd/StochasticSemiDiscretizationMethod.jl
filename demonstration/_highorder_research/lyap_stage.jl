# =============================================================================
# Stage-level COVARIANCE collocation for SCALAR linear SDDEs (d=1) — the correct engine.
#
# The augmented window y∈R^W evolves; its covariance C=E[y yᵀ]∈R^{W×W} obeys the window
# Lyapunov ODE. We collocate it like lyap_concept (which gave clean O(h^2S)), but build the
# window generator 𝓛 (on vec(C), W²×W²) from the SAME per-step structure as the first moment,
# with the noise term Σ 𝓖⊗𝓖 placed at the stage level (NOT deposited-then-propagated).
#
# Window first-moment generator 𝓐_w (W×W): we obtain it as the EXACT generator consistent
# with the collocation by: 𝓐_w = the block operator s.t. the newest block's endpoint rate is
# A·x + Σ B_k·x(t-τ_k) (delay read), the newest block's stage rates follow the collocation,
# and older blocks shift. Rather than hand-derive, we EXTRACT 𝓐_w per step from the verified
# per-step transition by 𝓐_w = (F̂ₙ − I)/h to leading order? NO — that is O(h). Instead we
# collocate the covariance with the implicit per-step Lyapunov solve directly (below).
#
# APPROACH (correct, mirrors lyap_concept's implicit stage solve, matrix-valued):
# Per step, treat the FULL window covariance vec(C)∈R^{W²} and apply ONE GL(S) collocation
# step of d vec(C)/dt = 𝓛(t) vec(C), where 𝓛(t) acts as:
#   reshape vec→C (W×W); compute Ċ = 𝓐_w C + C 𝓐_wᵀ + Σ_w 𝓖_w(t) C 𝓖_w(t)ᵀ ; vec back.
# We provide 𝓐_w, 𝓖_w as W×W matrices built from the window layout:
#   𝓖_w: deposits the noise coefficient α x + β x_τ (reads) into the newest block endpoint.
#   𝓐_w: newest-block endpoint rate = A x + B x_τ; PLUS the transport (older blocks ← newer).
# The transport part of 𝓐_w is the tricky bit; we get it RIGHT by matching the verified U:
#   require exp-collocation(𝓐_w, period) == U. We CALIBRATE the transport by building 𝓐_w
#   from the per-step block map's generator. For the SCALAR Hayes proof we use a DIRECT
#   construction of 𝓐_w on the (r+1) endpoint-only reduced window (q-Lagrange delay), which
#   is the classical method-of-steps companion generator — clean and exact.
# =============================================================================
using LinearAlgebra, Printf
include(joinpath(@__DIR__,"sdde_types.jl"))

# GL(S), S=1..6
function gl_tab(S)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5]
    elseif S==2; s3=sqrt(3); return [0.25 0.25-s3/6;0.25+s3/6 0.25],[0.5,0.5],[0.5-s3/6,0.5+s3/6]
    elseif S==3; s15=sqrt(15)
        return [5/36 2/9-s15/15 5/36-s15/30;5/36+s15/24 2/9 5/36-s15/24;5/36+s15/30 2/9+s15/15 5/36],[5/18,4/9,5/18],[0.5-s15/10,0.5,0.5+s15/10]
    else
        β=[k/sqrt(4k^2-1) for k in 1:S-1]; J=diagm(1=>β)+diagm(-1=>β); vals,vecs=eigen(Symmetric(J))
        c=(vals.+1)./2; b=vec(vecs[1,:]).^2
        V=[c[k]^(j-1) for k in 1:S,j in 1:S]; Aint=[c[i]^j/j for i in 1:S,j in 1:S]; a=Aint/V
        return a,b,c
    end
end

# GL(S) collocation monodromy for a LINEAR ODE d u/dt = 𝓛(t) u over [0,T], implicit stage
# solve (verified template from lyap_concept). 𝓛 returns D×D. Returns D×D monodromy.
function colloc_mono(𝓛, T, D, S, p)
    a,b,c=gl_tab(S); h=T/p; Φ=Matrix{Float64}(I,D,D)
    for n in 1:p
        t_n=(n-1)*h; Ls=[𝓛(t_n+c[i]*h) for i in 1:S]; SD=S*D
        M=Matrix{Float64}(I,SD,SD)
        for i in 1:S,j in 1:S; M[(i-1)*D+1:i*D,(j-1)*D+1:j*D]-=h*a[i,j]*Ls[j]; end
        RHS=zeros(SD,D); for i in 1:S; RHS[(i-1)*D+1:i*D,:]=Matrix(I,D,D); end
        Y=M\RHS; ynext=Matrix{Float64}(I,D,D)
        for j in 1:S; ynext+=h*b[j]*(Ls[j]*Y[(j-1)*D+1:j*D,:]); end
        Φ=ynext*Φ
    end
    return Φ
end

# =============================================================================
# Pseudospectral (Chebyshev) generator 𝓐_w for scalar autonomous DDE ẋ=a x+b x(t-τ).
# Represent the state on N+1 Chebyshev nodes on [-τ,0]. The infinitesimal generator of the
# DDE semigroup is the differentiation matrix D̃ with the top row replaced by the BC
# ẋ(0)=a x(0)+b x(-τ). Its eigenvalues → the DDE characteristic roots (high/spectral order).
# This is the EXACT high-order window generator we feed to the covariance Lyapunov.
# (Breda–Maset–Vermiglio pseudospectral approach.)
# =============================================================================
function cheb(N)   # Chebyshev nodes on [-1,1] and differentiation matrix (Trefethen)
    N==0 && return [0.0], reshape([0.0],1,1)
    x=[cos(pi*j/N) for j in 0:N]
    cc=[ (j==0||j==N) ? 2.0 : 1.0 for j in 0:N] .* [(-1.0)^j for j in 0:N]
    X=repeat(x,1,N+1); dX=X-X'
    Dm=(cc*(1.0./cc)')./(dX+Matrix(I,N+1,N+1)); Dm=Dm-diagm(vec(sum(Dm,dims=2)))
    return x,Dm
end

# window generator on N+1 nodes mapped to [-τ,0]; node 1 = θ=0 (present), node N+1 = θ=-τ.
function Aw_dde(a,b,τ,N)
    x,Dm=cheb(N)                       # x∈[-1,1], x[1]=1 (=θ=0), x[N+1]=-1 (=θ=-τ)
    D̃ = Dm .* (2/τ)                    # d/dθ on [-τ,0]
    A=copy(D̃)
    A[1,:] .= 0.0; A[1,1]=a; A[1,N+1]=b   # BC row: ẋ(0)=a x(node1) + b x(node N+1=−τ)
    return A                            # (N+1)×(N+1) window generator
end

# =============================================================================
# SECOND MOMENT via covariance Lyapunov on the pseudospectral window.
# State y = node values on [-τ,0] (N+1 nodes). First-moment generator 𝓐_w (Aw_dde).
# Noise generator 𝓖_w (per source): the multiplicative noise α x(0)+β x(-τ) injects into
# the PRESENT node's rate (node 1 = θ=0, the actual x). So 𝓖_w is (N+1)×(N+1) with
# 𝓖_w[1,1]=α, 𝓖_w[1,N+1]=β (rate of node 1 gets α·x(node1)+β·x(nodeN+1)); zero elsewhere
# (other nodes are pure history transport, no noise).
# Covariance C=E[y yᵀ] obeys dC/dt = 𝓐_w C + C 𝓐_wᵀ + Σ_j 𝓖_w,j C 𝓖_w,jᵀ (deterministic).
# We take its dominant eigenvalue over one delay τ via the matrix-exponential-free route:
# the generator on vec(C) is 𝓛 = I⊗𝓐_w + 𝓐_w⊗I + Σ_j 𝓖_w,j⊗𝓖_w,j ; multiplier over τ =
# dominant |exp(eig(𝓛)·τ)|.  (Spectral in N; exact Itô term — breaks the O(h³) wall.)
# =============================================================================
function Gw_dde(α,β,N)
    G=zeros(N+1,N+1); G[1,1]=α; G[1,N+1]=β; return G
end
# 2nd-moment dominant multiplier over τ for scalar SDDE ẋ=(a x+b x(-τ))dt+(α x+β x(-τ))dW.
function rho2_pseudo(a,b,α,β,τ,N)
    A=Aw_dde(a,b,τ,N); G=Gw_dde(α,β,N); M=N+1; Id=Matrix(I,M,M)
    𝓛 = kron(Id,A) + kron(A,Id) + kron(G,G)        # generator on vec(C), (M²)×(M²)
    ev=eigen(𝓛).values
    # dominant 2nd-moment multiplier over one delay τ
    return maximum(abs.(exp.(ev .* τ)))
end
rho1_pseudo(a,b,τ,N)=(A=Aw_dde(a,b,τ,N); ev=eigen(A).values; maximum(abs.(exp.(ev.*τ))))

println("lyap_stage loaded")

function _run_tests()
    a=-1.0; b=-0.4; τ=1.0
    println("=== pseudospectral FIRST-moment check (det. ẋ=ax+bx(t-τ)) ===")
    fch(λ)=λ-a-b*exp(-λ*τ); fp(λ)=1+b*τ*exp(-λ*τ); λ=complex(a+b)
    for _ in 1:200; λ=λ-fch(λ)/fp(λ); end
    multexact=abs(exp(λ*τ)); @printf("  exact |exp(λτ)| = %.10f\n", multexact)
    for N in [4,8,12,16,20]
        @printf("  N=%2d  ρ1=%.10f  err=%.2e\n", N, rho1_pseudo(a,b,τ,N), abs(rho1_pseudo(a,b,τ,N)-multexact))
    end

    println("\n=== Hayes 2nd moment (β=0.3 delayed noise) → 0.57022372583 ===")
    ref=0.57022372583
    for N in [4,8,12,16,20,24]
        ρ=rho2_pseudo(a,b,0.0,0.3,τ,N)
        @printf("  N=%2d  ρ2=%.11f  err=%.2e\n", N, ρ, abs(ρ-ref))
    end

    println("\n=== scalar PRESENT-noise dx=αx dW (no delay b=β=0) → exp((2a+α²)τ) ===")
    aa=-0.7; α=0.5
    ex=exp((2aa+α^2)*τ); @printf("  exact=%.11f\n", ex)
    for N in [2,4,6,8,12]
        ρ=rho2_pseudo(aa,0.0,α,0.0,τ,N)
        @printf("  N=%2d  ρ2=%.11f  err=%.2e\n", N, ρ, abs(ρ-ex))
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    _run_tests()
end
