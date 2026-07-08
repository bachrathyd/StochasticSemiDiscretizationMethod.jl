# Work-precision figure: accuracy (error in ρ(H)) vs CPU time for the high-order
# Gauss–Legendre collocation (GL1/GL2/GL3) against the classical semi-discretization.
# This is where the collocation solver shines: high order ⇒ many digits at a
# fraction of the cost. Saves assets/HighOrderConvergence.png.
#   julia --project=examples examples/plot_highorder_wp.jl
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod, StaticArrays, Plots, Printf
gr()

# delayed-PD Mathieu (β ≡ 0 ⇒ fast pruned collocation), present-state noise + additive
prob = LDDEProblem(
    ProportionalMX(t -> @SMatrix [0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]),
    [DelayMX(1.0, t -> @SMatrix [0.0 0.0; 0.20*(1+0.3cos(2π*t)) 0.12*(1+0.4cos(2π*t))])],
    [stCoeffMX(1, ProportionalMX(t -> @SMatrix [0.0 0.0; 0.30 0.0]))],
    [stCoeffMX(1, DelayMX(1.0, t -> @SMatrix zeros(2,2)))],
    Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
const T = 1.0
ρref = spectralRadiusOfMoment(prob, T, 160; S=3)          # fine reference
@printf("reference ρ(H) = %.14f\n", ρref)

tmin(f) = (f(); minimum(@elapsed(f()) for _ in 1:2))       # warm, then best-of-2

series = [
    ("classical SD (order 1)", ClassicalSD(2), (8,16,24,32,48,64,96,128), :gray45,   :circle),
    ("GL1  (order 2)",         GaussLegendre(1), (4,6,8,12,16,24,32),      :seagreen, :utriangle),
    ("GL2  (order 4)",         GaussLegendre(2), (4,6,8,12,16,24),         :dodgerblue,:diamond),
    ("GL3  (order 6)",         GaussLegendre(3), (4,6,8,12,16),            :crimson,  :star5),
]

plt = plot(; xscale=:log10, yscale=:log10, xlabel="CPU time [s]",
           ylabel="error in ρ(ℋ)", legend=:bottomleft, framestyle=:box,
           minorgrid=true, gridalpha=0.3, size=(640,460), dpi=150,
           title="Second-moment stability — accuracy vs cost")
for (lab, method, ps, col, mk) in series
    ts = Float64[]; es = Float64[]
    for p in ps
        e = abs(spectralRadiusOfMoment(prob, T, p; method=method) - ρref)
        e < 1e-13 && continue                                 # below the solver floor
        push!(ts, tmin(() -> spectralRadiusOfMoment(prob, T, p; method=method)))
        push!(es, e)
    end
    plot!(plt, ts, es; label=lab, color=col, marker=mk, markersize=5, lw=2, markerstrokewidth=0)
    @printf("%-22s: %d points, error %.1e → %.1e\n", lab, length(es), es[1], es[end])
end
savefig(plt, joinpath(@__DIR__, "..", "assets", "HighOrderConvergence.png"))
println("saved assets/HighOrderConvergence.png")
