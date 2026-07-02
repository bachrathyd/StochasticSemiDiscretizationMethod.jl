# Validate moment-DDE engine on the 2D PERIODIC stochastic Mathieu (present-state α noise)
# against the trusted SDM ρ(H). Period P=4π, delay τ=2π (so T>τ, r=p/2).
using Printf
include(joinpath(@__DIR__,"moment_engine.jl"))
include(joinpath(@__DIR__,"moment_engine2.jl"))

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1

# x'' + 2ζx' + (A+ε cos(t/2))x = B x(t-τ) + noise.  d=2.
prob = SDDEProblem(
    2, PER,
    t -> [0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA],
    [(t->TAU, t->[0.0 0.0; Bval 0.0])],
    [ ( t->[0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA],   # present-state α
        [t->[0.0 0.0; ALPHA*Bval 0.0]],                            # delayed β
        t->[0.0,0.0] ) ]
)

const REF = 0.156228322806   # trusted SDM order-2 high-p (from earlier session)
println("Validate Mathieu moment-engine vs trusted SDM ρ(H)≈$REF\n")
for S in 1:3
    @printf("GL(%d):\n", S)
    prev=nothing
    for p in [20,40,80,160]
        ρ = second_moment_rho(prob, S, p)
        err=abs(ρ-REF)
        rate = prev===nothing ? NaN : log2(prev/err)
        @printf("  p=%3d  ρ=%.10f  err=%.2e  rate=%.2f\n", p, ρ, err, rate)
        prev=err
    end
    println()
end
