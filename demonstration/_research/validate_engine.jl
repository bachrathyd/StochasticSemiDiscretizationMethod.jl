# Validate the general moment engine against the PROVEN scalar Hayes result.
# Hayes: dx=(A x + B x(t-1))dt + (β x(t-1)) dW,  A=-1,B=-0.4,β=0.3, τ=1.
# Proven dominant 2nd-moment multiplier (scratch_delay_moment.jl): 0.57022372583
# Expect: GL1 O(h²), GL2 O(h⁴), GL3 O(h⁶).
using Printf
include(joinpath(@__DIR__,"moment_engine.jl"))
include(joinpath(@__DIR__,"moment_engine2.jl"))

const A=-1.0; const B=-0.4; const BETA=0.3

prob = SDDEProblem(
    1, 1.0,                                  # d=1, period T=1 (=τ here, so r=p)
    t -> [A;;],                              # A(t)
    [(t->1.0, t->[B;;])],                    # one delay τ=1, B
    [ (t->[0.0;;], [t->[BETA;;]], t->[0.0]) ]  # one noise: α=0, β=BETA (delayed mult.), σ=0
)

const REF = 0.57022372583
println("Validate moment engine vs proven Hayes ρ=$REF\n")
for S in 1:3
    @printf("GL(%d):\n", S)
    prev=nothing
    for p in [4,8,16,32]
        ρ = second_moment_rho(prob, S, p)
        err=abs(ρ-REF)
        rate = prev===nothing ? NaN : log2(prev/err)
        @printf("  p=%2d  ρ=%.11f  err=%.2e  rate=%.2f\n", p, ρ, err, rate)
        prev=err
    end
    println()
end
