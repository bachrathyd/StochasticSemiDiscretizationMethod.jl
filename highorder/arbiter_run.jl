# =============================================================================
# Arbiter validation + arbitration of the three disputed values.
#
# Gates (arbiter must earn trust first):
#   A-1 power iteration == dense vech eigen at tiny N
#   A-2 noise-off ⇒ ρ(H) → ρ(U)²  (ρU from the verified deterministic GL3)
#   A-3 present-noise no-delay ⇒ exp((2a+α²)T), O(h²) + Richardson hits it
# Arbitration:
#   B-1 mirror Mathieu:   v7-family says 0.7389661254; SDM-q2 (measured-order-1
#       extrapolation) also → ~0.738966; exponent-3 Richardson said 0.7389577.
#   B-2 Hayes delayed noise (A=-1,B=-0.4,β=0.3): trusted ≈ 0.148x
#   B-3 critical Mathieu: archived unresolved gap 0.15622747 (v6 GL4) vs
#       0.15622870 (old SDM-q2-Richardson).
# =============================================================================
include(joinpath(@__DIR__, "arbiter_finegrid.jl"))
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))
using Printf

println("── A-1: power iteration vs dense (mirror problem, N=16) ──")
pb_mirror_fg = FGProb(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.2 0.0],
    t->[0.0 0.0; 0.2 0.0],
    t->[0.0 0.0; 0.2 0.0])
ρp = fg_rho_H(pb_mirror_fg, 16)
ρd = fg_rho_H_dense(pb_mirror_fg, 16)
@printf("  power=%.12f dense=%.12f rel=%.1e %s\n", ρp, ρd, abs(ρp-ρd)/ρd,
        abs(ρp-ρd)/ρd < 1e-8 ? "PASS" : "FAIL")

println("── A-2: noise-off gate (B=0.2, α=β=0) ──")
# NOTE: with noise OFF the deterministic monodromy has a dominant COMPLEX pair,
# so ρ(H) is attained by a non-simple eigenvalue triple (λ², λ̄², |λ|²) of equal
# modulus — plain power iteration oscillates there. The Itô source restores a
# strictly dominant Perron eigenvalue in every with-noise case, where power
# iteration is used. The noise-off gate therefore uses the DENSE eigensolve.
pb_off_fg = FGProb(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.2 0.0],
    t->zeros(2,2), t->zeros(2,2))
pb_off_v7 = Prob(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.2 0.0],
    t->zeros(2,2), t->zeros(2,2))
ρU2 = rho_U_v7(build_v7(pb_off_v7, 3, 48))^2
@printf("  target ρU² (GL3 p=48) = %.12f\n", ρU2)
errs_off=Float64[]
for N in (16, 32, 64)
    ρN = fg_rho_H_dense(pb_off_fg, N)
    push!(errs_off, abs(ρN-ρU2))
    @printf("  fg-dense N=%4d ρ=%.12f err=%.2e\n", N, ρN, errs_off[end])
end
println("  rates: ", join([@sprintf("%.2f",log2(errs_off[i]/errs_off[i+1])) for i in 1:2], ", "),
        "  (expect ≈2 → gate PASS iff h² convergence to ρU²)")

println("── A-3: present-noise exact value exp((2a+α²)T) ──")
a_c=-0.8; α_c=0.4
ρ_exact = exp(2a_c+α_c^2)
pb_pres = FGProb(1, 1.0, 0.25,
    t->fill(a_c,1,1), t->zeros(1,1), t->fill(α_c,1,1), t->zeros(1,1))
errs=Float64[]
for N in (64, 128, 256, 512)
    ρN = fg_rho_H(pb_pres, N)
    push!(errs, abs(ρN-ρ_exact))
    @printf("  fg N=%4d ρ=%.12f err=%.2e\n", N, ρN, errs[end])
end
rates=[log2(errs[i]/errs[i+1]) for i in 1:length(errs)-1]
println("  rates: ", join([@sprintf("%.2f",r) for r in rates], ", "), "  (expect ≈2)")
ρA3 = fg_arbiter(pb_pres, [128,256,512])
@printf("  Richardson vs exact: err=%.2e %s\n", abs(ρA3-ρ_exact),
        abs(ρA3-ρ_exact)<1e-8 ? "PASS" : "FAIL")

println("── B-1: ARBITRATE mirror Mathieu ──")
println("  v7 causal GL3 (converged): 0.7389661254")
ρB1 = fg_arbiter(pb_mirror_fg, [256, 512, 1024, 2048])
@printf("  arbiter says %.10f;  |arb − v7| = %.2e\n", ρB1, abs(ρB1-0.7389661254))

println("── B-2: ARBITRATE Hayes delayed noise ──")
pb_hayes_fg = FGProb(1, 1.0, 1.0,
    t->fill(-1.0,1,1), t->fill(-0.4,1,1), t->zeros(1,1), t->fill(0.3,1,1))
ρB2 = fg_arbiter(pb_hayes_fg, [256, 512, 1024, 2048])
@printf("  arbiter says %.10f\n", ρB2)

println("── B-3: ARBITRATE critical stochastic Mathieu ──")
pb_math_fg = FGProb(2, 4π, 2π,
    t->[0.0 1.0; -(3.0+2.0*cos(0.5*t)) -0.2],
    t->[0.0 0.0; 0.5 0.0],
    t->[0.0 0.0; -0.1*(3.0+2.0*cos(0.5*t)) -0.1*0.2],
    t->[0.0 0.0; 0.1*0.5 0.0])
ρB3 = fg_arbiter(pb_math_fg, [256, 512, 1024, 2048])
@printf("  arbiter says %.10f;  v6GL4=0.15622747, oldSDMRich=0.15622870\n", ρB3)
println("done")
