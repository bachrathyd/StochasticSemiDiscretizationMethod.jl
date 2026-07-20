# Work-precision figure: accuracy (error in ρ(H)) vs CPU time for the high-order
# Gauss–Legendre collocation (GL1..GL5) against the classical semi-discretization,
# on a CRITICAL problem — a lightly-damped delayed Mathieu oscillator with a high
# natural frequency (ωn=5) whose delay spans ~5 oscillations, so many
# discretization points are needed and the full convergence onset (from p=1) is
# visible. Timing uses BenchmarkTools (minimum of a few samples) for clean,
# reproducible CPU costs. Saves assets/HighOrderConvergence.png.
#   julia --project=examples examples/plot_highorder_wp.jl
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod, StaticArrays, Plots, Printf, LinearAlgebra, BenchmarkTools
gr()

# Pin BLAS to a single thread — these solvers do many small/thin matrix products
# and multithreaded OpenBLAS adds fork/join overhead (a jump in CPU time at a
# size threshold). One thread is smoother and faster; parallelise chart sweeps
# over the outer parameter loop instead.
BLAS.set_num_threads(1)
@printf("BLAS threads = %d\n", BLAS.get_num_threads())

# Critical delayed Mathieu: ẍ + 2ζωn ẋ + ωn²(1+ε cos t) x = b x(t−τ) + noise,
# ωn=5, ζ=0.05, ε=0.3, τ = T = 2π ⇒ the delay spans ~ωn·τ/2π = 5 oscillations.
# β ≡ 0 ⇒ fast pruned collocation; present-state multiplicative noise + additive.
const ωn = 5.0; const ζc = 0.05; const εc = 0.3
const prob = LDDEProblem(
    ProportionalMX(t -> @SMatrix [0.0 1.0; -(ωn^2*(1+εc*cos(t))) -2ζc*ωn]),
    [DelayMX(2π, t -> @SMatrix [0.0 0.0; 0.30 0.0])],
    [stCoeffMX(1, ProportionalMX(t -> @SMatrix [0.0 0.0; 0.10 0.0]))],
    [stCoeffMX(1, DelayMX(2π, t -> @SMatrix zeros(2,2)))],
    Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
const T = 2π
const TOL = 1e-13                                          # tight ⇒ low error floor
ρval(p, method) = spectralRadiusOfMoment(prob, T, p; method=method, tol=TOL)
const ρref = spectralRadiusOfMoment(prob, T, 96; method=GaussLegendre(5), tol=1e-14)
@printf("reference ρ(H) = %.14f\n", ρref)

# clean CPU time via BenchmarkTools: minimum of a handful of samples, total
# benchmarking capped at ~1.5 s per point (few evaluations, robust to jitter).
function cputime(method, p)
    b = @benchmark spectralRadiusOfMoment($prob, $T, $p; method=$method, tol=$TOL) samples=6 seconds=1.5 evals=1
    minimum(b.times) / 1e9                                  # seconds
end

const TCAP  = 1.0                                          # extend p until ~1 s CPU/point
const FLOOR = 3e-13                                        # drop points at the solver floor
# ~3× denser than before: log-spaced p = 1 … 1000 (deduped integers)
const PS = sort(unique(round.(Int, 10 .^ range(0.0, 3.0, length=34))))

series = [
    ("classical SD (order 1)", ClassicalSD(2), :gray45,     :circle),
    ("GL1  (order 2)",  GaussLegendre(1),      :seagreen,   :utriangle),
    ("GL2  (order 4)",  GaussLegendre(2),      :dodgerblue, :diamond),
    ("GL3  (order 6)",  GaussLegendre(3),      :crimson,    :star5),
    ("GL4  (order 8)",  GaussLegendre(4),      :darkorange, :hexagon),
    ("GL5  (order 10)", GaussLegendre(5),      :purple,     :pentagon),
]

plt = plot(; xscale=:log10, yscale=:log10, xlabel="CPU time [s]",
           ylabel="error in ρ(ℋ)", legend=:bottomleft, framestyle=:box,
           minorgrid=true, gridalpha=0.25, size=(820,600), dpi=150,
           legendfontsize=8, title="Second-moment stability — accuracy vs cost (critical ωn=5)")
for (lab, method, col, mk) in series
    ts = Float64[]; es = Float64[]
    for p in PS
        e = abs(ρval(p, method) - ρref)
        (isfinite(e) && e ≥ FLOOR) || continue             # skip NaN / below-floor
        t = cputime(method, p)
        push!(ts, t); push!(es, e)
        t > TCAP && break                                  # cap CPU time per resolution
    end
    isempty(es) && (@printf("%-22s: (no points above floor)\n", lab); continue)
    plot!(plt, ts, es; label=lab, color=col, marker=mk, markersize=4,
          markeralpha=0.85, lw=1.7, markerstrokewidth=0)
    @printf("%-22s: %2d pts, err %.1e → %.1e, CPU %.5f → %.4f s\n",
            lab, length(es), es[1], es[end], ts[1], ts[end])
end
savefig(plt, joinpath(@__DIR__, "..", "assets", "HighOrderConvergence.png"))
println("saved assets/HighOrderConvergence.png")
