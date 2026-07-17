# Journal figure: reproduction of Sykora & Bachrathy (2020) Fig. 4 — 2nd-moment
# stability boundaries of the stochastic delayed Mathieu equation in the (A,B)
# plane for increasing multiplicative noise strength. Journal styling matches
# the other figures of the paper. MDBM boundary detection, q=2 discretization.
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, MDBM, Plots, Printf, LinearAlgebra
BLAS.set_num_threads(1)

function createStochMathieuProblem(A, ε, B, ζ, τ, α_val)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε * cos(0.5 * t)) -2ζ]
    αMxfun(t) = @SMatrix [0. 0.; -α_val*(A + ε*cos(0.5*t)) -α_val*2ζ]
    LDDEProblem(ProportionalMX(AMxfun), [DelayMX(τ, @SMatrix [0. 0.; B 0.])],
        [stCoeffMX(1, ProportionalMX(αMxfun))],
        [stCoeffMX(1, DelayMX(τ, @SMatrix [0. 0.; α_val*B 0.]))],
        Additive(2), [stAdditive(1, Additive(@SVector [0., 0.]))])
end

const ε=2.0; const ζ=0.1; const τ=2π; const P=4π; const p_res=40
const α_vals = [0.0, 0.1, 0.2, 0.3]
cols = [:black, :dodgerblue, :darkorange, :crimson]

plt = plot(xlabel="A", ylabel="B", xlim=(0,5), ylim=(-1.5,1.5),
           size=(900,620), framestyle=:box, dpi=300,
           guidefontsize=13, tickfontsize=11, legendfontsize=10,
           left_margin=5Plots.mm, bottom_margin=5Plots.mm,
           legend=:topright, minorgrid=true, gridalpha=0.25, minorgridalpha=0.10)

for (i,α_val) in enumerate(α_vals)
    method = SemiDiscretization(2, P/p_res)
    foo(A::Float64, B::Float64)::Float64 = log(spectralRadiusOfMapping_MF_factored(
        SSDM.calculateResults(createStochMathieuProblem(A,ε,B,ζ,τ,α_val), method, τ, n_steps=p_res)))
    @printf("boundary for α=%.1f ...\n", α_val); flush(stdout)
    mdbm = MDBM_Problem(foo, [Axis(0.0:0.25:5.0,:A), Axis(-1.5:0.25:1.5,:B)])
    solve!(mdbm, 4, verbosity=0, doThreadprecomp=false)
    pts = getinterpolatedsolution(mdbm)
    scatter!(plt, pts..., markersize=2.2, markerstrokewidth=0, color=cols[i],
             label="α = $(α_val)")
    savefig(plt, joinpath(@__DIR__,"..","assets","sykora_fig4_pub.png"))  # live
end
savefig(plt, joinpath(@__DIR__,"..","assets","sykora_fig4_pub.png"))
savefig(plt, joinpath(@__DIR__,"..","assets","sykora_fig4_pub.pdf"))
dst = joinpath(@__DIR__, "..", "journal_paper", "images"); mkpath(dst)  # local paper images
cp(joinpath(@__DIR__,"..","assets","sykora_fig4_pub.png"), joinpath(dst,"fig1_sykora_fig4_repro.png"); force=true)
cp(joinpath(@__DIR__,"..","assets","sykora_fig4_pub.pdf"), joinpath(dst,"fig1_sykora_fig4_repro.pdf"); force=true)
println("done — sykora fig4 restyled")
