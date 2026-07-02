# Generalize the moment-DDE to include PRESENT-state multiplicative noise α (Mathieu has it).
# Scalar:  dx = (A x + B x_{t-1}) dt + (α x + β x_{t-1}) dW.
# Itô moment system (M=E[x²], P=E[x_t x_{t-1}], N=M(t-1)):
#   d(x²) = 2x dx + (dx)²;  (dx)² = (α x + β x_{t-1})² dt
#   dM/dt = 2A M + 2B P + α² M + 2αβ P + β² N
#         = (2A+α²) M + (2B+2αβ) P + β² M(t-1)
#   dP/dt = E[dx · x_{t-1}] = A P + B N           (x_{t-1} indep. of dW_t ⇒ no Itô cross term)
#         = A P + B M(t-1)
# Delay system u=(M,P): u'(t) = A0 u(t) + A1 u(t-1), with A0,A1 below.
# Validate GL(S) superconvergence vs high-p reference, and sanity-check the α² Itô term by
# comparing the NO-DELAY limit (B=β=0) to exact exp((2A+α²)T).
using LinearAlgebra, SparseArrays, Printf

const A_t=-1.0; const B_t=-0.4; const alf=0.3; const bet=0.2; const TAU=1.0

const A0 = [2A_t+alf^2   2B_t+2*alf*bet;
            0.0           A_t]
const A1 = [bet^2  0.0;
            B_t     0.0]

include(joinpath(@__DIR__,"moment_colloc.jl"))   # GL collocation window monodromy (D=2)

ref = rho_moment(A0,A1,3,256,TAU)
@printf("Reference (GL3 p=256) dominant 2nd-moment multiplier = %.12f\n\n", ref)
for S in 1:3
    @printf("GL(%d) on full moment-DDE (with α present-noise):\n", S)
    prev=nothing
    for p in [4,8,16,32,64]
        ρ=rho_moment(A0,A1,S,p,TAU); err=abs(ρ-ref)
        rate=prev===nothing ? NaN : log2(prev/err)
        @printf("  p=%3d  ρ=%.11f  err=%.2e  rate=%.2f\n",p,ρ,err,rate); prev=err
    end
    println()
end

# No-delay sanity: B=β=0 ⇒ exact multiplier exp((2A+α²)T)
println("No-delay sanity (B=β=0): exact = ", exp((2A_t+alf^2)*TAU))
A0n=[2A_t+alf^2 0.0; 0.0 A_t]; A1n=zeros(2,2)
for S in 1:3
    ρ=rho_moment(A0n,A1n,S,64,TAU)
    @printf("  GL(%d) p=64  ρ=%.11f  err=%.2e\n",S,ρ,abs(ρ-exp((2A_t+alf^2)*TAU)))
end
