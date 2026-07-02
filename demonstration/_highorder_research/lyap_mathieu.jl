# =============================================================================
# Periodic Mathieu (d=2) high-order 2nd moment via pseudospectral window + time-dependent
# covariance Lyapunov, integrated over the period (Floquet).
#
# State x∈R² (x, ẋ). DDE history on N+1 Chebyshev nodes over [-τ,0] → window y∈R^{(N+1)·2}.
# Window generator 𝓐_w(t): block differentiation (transport) with the BC row (node θ=0):
#   ẋ(t) = A(t) x(t) + B x(t-τ)       (the Mathieu ODE; node 1 = present, node N+1 = -τ).
# Noise generator 𝓖_w(t) (one source): present α(t) on node1, delayed β on node(N+1),
#   injected into node1's rate (the actual ẋ equation carries the noise).
# Covariance vec(C) (dim ((N+1)·2)²) obeys d vec(C)/dt = 𝓛(t) vec(C), periodic. The period
# monodromy (collocated, GL(S)) has ρ = ρ(H) — high order in p AND spectral in N.
#
# Compare to trusted SDM q=2 Richardson ≈ 0.156228.
# =============================================================================
using LinearAlgebra, Printf

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1
Amat(t)=[0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA]
Bmat()=[0.0 0.0; Bval 0.0]
αmat(t)=[0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA]   # present-state mult. noise
βmat()=[0.0 0.0; ALPHA*Bval 0.0]                               # delayed mult. noise

# Chebyshev nodes/diff matrix on [-1,1]
function cheb(N)
    N==0 && return [0.0],reshape([0.0],1,1)
    x=[cos(pi*j/N) for j in 0:N]
    cc=[(j==0||j==N) ? 2.0 : 1.0 for j in 0:N].*[(-1.0)^j for j in 0:N]
    X=repeat(x,1,N+1); dX=X-X'
    D=(cc*(1.0./cc)')./(dX+Matrix(I,N+1,N+1)); D=D-diagm(vec(sum(D,dims=2)))
    return x,D
end

# window generators for d=2 on (N+1) nodes (node1=θ=0 present, node N+1=θ=-τ).
# returns functions of t for 𝓐_w and 𝓖_w (each ((N+1)·2)×((N+1)·2)).
function build_mathieu_window(N)
    x,Dm=cheb(N); D̃=Dm.*(2/TAU)                 # d/dθ on [-τ,0]
    M=N+1; d=2; W=M*d
    # transport part: kron(D̃, I_2) acts on stacked node-blocks [x(node1);...;x(nodeM)]
    Dblk=kron(D̃,Matrix(I,d,d))
    function Aw(t)
        A=copy(Dblk)
        # BC: node1 rate = A(t) x(node1) + B x(nodeM)  (overwrite node1's d rows)
        A[1:d,:] .= 0.0
        A[1:d,1:d] = Amat(t)
        A[1:d,(M-1)*d+1:M*d] = Bmat()
        return A
    end
    function Gw(t)
        G=zeros(W,W)
        G[1:d,1:d] = αmat(t)                       # present noise on node1
        G[1:d,(M-1)*d+1:M*d] = βmat()              # delayed noise from nodeM
        return G
    end
    return Aw,Gw,W
end

# GL(S) tableau
function gl_tab(S)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5] end
    β=[k/sqrt(4k^2-1) for k in 1:S-1]; J=diagm(1=>β)+diagm(-1=>β); E=eigen(Symmetric(J))
    c=(E.values.+1)./2; b=vec(E.vectors[1,:]).^2
    Vm=[c[k]^(j-1) for k in 1:S,j in 1:S]; Aint=[c[i]^j/j for i in 1:S,j in 1:S]
    return Aint/Vm, b, c
end

# collocate d vec(C)/dt = 𝓛(t) vec(C) over [0,PER], GL(S), p steps; ρ(monodromy).
# 𝓛(t) v = vec( Aw(t) C + C Aw(t)ᵀ + Gw(t) C Gw(t)ᵀ ),  C=reshape(v,W,W).
function rho2_mathieu(N,S,p)
    Aw,Gw,W=build_mathieu_window(N); D=W*W; a,b,c=gl_tab(S); h=PER/p
    Lop(t)=begin
        A=Aw(t); G=Gw(t); Id=Matrix(I,W,W)
        kron(Id,A)+kron(conj(A),Id)+kron(conj(G),G)      # vec(AC+CAᵀ+GCGᵀ) operator (real)
    end
    # but building D×D=W⁴ is huge; instead apply 𝓛 matrix-free in the collocation solve.
    # For the PROOF (small N), W is modest; W=(N+1)*2. N=6→W=14→D=196→D²=38k entries, ok dense.
    Φ=Matrix{Float64}(I,D,D)
    for n in 1:p
        t_n=(n-1)*h; Ls=[Lop(t_n+c[i]*h) for i in 1:S]; SD=S*D
        M=Matrix{Float64}(I,SD,SD)
        for i in 1:S,j in 1:S; M[(i-1)*D+1:i*D,(j-1)*D+1:j*D]-=h*a[i,j]*Ls[j]; end
        RHS=zeros(SD,D); for i in 1:S; RHS[(i-1)*D+1:i*D,:]=Matrix(I,D,D); end
        Y=M\RHS; yn=Matrix{Float64}(I,D,D)
        for j in 1:S; yn+=h*b[j]*(Ls[j]*Y[(j-1)*D+1:j*D,:]); end
        Φ=yn*Φ
    end
    return maximum(abs.(eigen(Φ).values))
end

if abspath(PROGRAM_FILE)==@__FILE__
    const REF=0.156228322806
    println("Periodic Mathieu d=2, 2nd moment ρ(H); trusted SDM ref ≈ ", REF, "\n")
    for N in [4,6,8]
        @printf("N=%d (pseudospectral window):\n", N)
        for S in [2,3]
            for p in [8,16,32]
                ρ=rho2_mathieu(N,S,p)
                @printf("   GL%d p=%2d  ρ=%.10f  err=%.2e\n", S, p, ρ, abs(ρ-REF))
            end
        end
    end
end
