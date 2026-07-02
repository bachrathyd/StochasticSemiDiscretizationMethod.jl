# Finer arbitration of the critical stochastic Mathieu (drift was 6.8e-6 at
# N≤2048) + v7 causal GL3/GL4 at higher p for the same target.
include(joinpath(@__DIR__, "arbiter_finegrid.jl"))
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))
using Printf

pb_fg = FGProb(2, 4π, 2π,
    t->[0.0 1.0; -(3.0+2.0*cos(0.5*t)) -0.2],
    t->[0.0 0.0; 0.5 0.0],
    t->[0.0 0.0; -0.1*(3.0+2.0*cos(0.5*t)) -0.1*0.2],
    t->[0.0 0.0; 0.1*0.5 0.0])
println("── arbiter, critical Mathieu, up to N=8192 ──")
ρarb = fg_arbiter(pb_fg, [1024, 2048, 4096, 8192])

println("── v7 causal GL3/GL4 up to p=128 ──")
pb = Prob(2, 4π, 2π,
    t->[0.0 1.0; -(3.0+2.0*cos(0.5*t)) -0.2],
    t->[0.0 0.0; 0.5 0.0],
    t->[0.0 0.0; -0.1*(3.0+2.0*cos(0.5*t)) -0.1*0.2],
    t->[0.0 0.0; 0.1*0.5 0.0])
for S in (3, 4), p in (48, 64, 96, 128)
    t0=time()
    ρ = rho_H_krylov(build_v7(pb, S, p); offdiag=:causal)
    @printf("  GL%d p=%3d ρ=%.10f  |ρ−arb|=%.2e  (%.0fs)\n", S, p, ρ, abs(ρ-ρarb), time()-t0)
    flush(stdout)
end
println("archived candidates: v6GL4p120=0.15622747, oldSDMRich=0.15622870")
