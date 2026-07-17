# CPU comparison on the Iklodi-Dankowicz turning benchmark (their Fig. 13,
# model Eq. 94-95, ζ=0.05, σ=1). We trace the SAME second-moment stability
# boundary ρ(H)=1 with MF-SSDM + MDBM and time it, to compare against their
# quoted costs: pseudo-spectral 6 s / 2 min / 30 min / 22 h at M=5/10/20/40,
# and 10 s for their discretization-free continuation.
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod, StaticArrays, LinearAlgebra, Printf, BenchmarkTools
const SSDM = StochasticSemiDiscretizationMethod
BLAS.set_num_threads(1)
using MDBM

const ζ = 0.05; const σ = 1.0
const RRES = 24; const NATRES = 30      # ≥24 steps/delay, ≥30 steps/natural period

# autonomous 2-D turning SDDE (Eq. 95) as a τ-periodic LDDEProblem; τ = 2π/ω̃
function turning_prob(ω̃, w̃)
    τ = 2π/ω̃
    A(t) = @SMatrix [0.0 1.0; -1.0-w̃ -2ζ]
    B(t) = @SMatrix [0.0 0.0; w̃ 0.0]
    a(t) = @SMatrix [0.0 0.0; -σ*w̃ 0.0]        # present multiplicative noise (α)
    b(t) = @SMatrix [0.0 0.0;  σ*w̃ 0.0]        # delayed multiplicative noise (β ≠ 0)
    LDDEProblem(ProportionalMX(A), [DelayMX(τ, B)],
        [stCoeffMX(1, ProportionalMX(a))], [stCoeffMX(1, DelayMX(τ, b))],
        Additive(2), [stAdditive(1, Additive(@SVector [0.0, σ*w̃]))])
end
function rho_sm(ω̃, w̃)
    τ = 2π/ω̃
    Δt = min(τ/RRES, 2π/NATRES); p = max(1, Int(round(τ/Δt)))
    rst = SSDM.calculateResults(turning_prob(ω̃, w̃), SemiDiscretization(2, τ/p), τ; n_steps=p)
    spectralRadiusOfMapping_MF_factored(rst)
end

# sanity at their labelled points (A stable, B unstable)
@printf("point A (ω̃=1.00,w̃=0.10): ρ(H)=%.4f  %s\n", rho_sm(1.0,0.10), rho_sm(1.0,0.10)<1 ? "stable ✓" : "UNSTABLE")
@printf("point B (ω̃=1.00,w̃=0.55): ρ(H)=%.4f  %s\n", rho_sm(1.0,0.55), rho_sm(1.0,0.55)<1 ? "STABLE" : "unstable ✓")

# per-point cost at the most expensive resolution (low speed ⇒ large delay ⇒ large p)
let ω̃=0.15
    τ=2π/ω̃; Δt=min(τ/RRES,2π/NATRES); p=Int(round(τ/Δt))
    tp = @belapsed rho_sm($ω̃, 0.3) samples=8 seconds=3
    @printf("per-point cost at ω̃=%.2f (τ=%.1f, p=%d):  %.4f s\n", ω̃, τ, p, tp)
end

# full second-moment lobe boundary ρ(H)=1 over the Fig-13 window, timed
f(ω̃, w̃) = rho_sm(ω̃, w̃) - 1.0
ax = [Axis(range(0.12, 2.0, length=24), :ω̃), Axis(range(0.0, 0.8, length=10), :w̃)]
prob = MDBM_Problem(f, ax)
GC.gc()
tlobe = @elapsed begin
    solve!(prob, 4; interpolationorder=0)     # bracketing refinements
    solve!(prob, 0; interpolationorder=1)     # one linear-interp pass
end
pts = getinterpolatedsolution(prob)
@printf("\n=== FULL second-moment lobe: %d boundary points traced in %.2f s (1 core) ===\n", length(pts[1]), tlobe)
open(joinpath(@__DIR__,"cmp_turning_lobe.csv"),"w") do io
    println(io,"omega_tilde,w_tilde")
    for k in eachindex(pts[1]); @printf(io,"%.6f,%.6f\n",pts[1][k],pts[2][k]); end
end
println("done — cmp_turning_lobe.csv")
