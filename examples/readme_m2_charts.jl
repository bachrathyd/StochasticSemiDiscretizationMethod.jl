# Regenerates the README's two 2nd-moment stability-border charts
# (assets/StochHayesM2.png, assets/StochLDOM2.png) from the README code blocks.
#   julia --project=examples examples/readme_m2_charts.jl
using StochasticSemiDiscretizationMethod, StaticArrays, MDBM, Plots, LaTeXStrings
using LinearAlgebra; BLAS.set_num_threads(1)
gr()
const ASSETS = joinpath(@__DIR__, "..", "assets")

# ---- Hayes 2nd-moment stability border ----
function createHayesProblem(a, β)
    AMx = ProportionalMX(a*ones(1,1)); τ1 = 1.0
    BMx1 = DelayMX(τ1, zeros(1,1))
    cVec = Additive(1); noiseID = 1
    αMx1 = stCoeffMX(noiseID, ProportionalMX(zeros(1,1)))
    βMx11 = stCoeffMX(noiseID, DelayMX(τ1, β*ones(1,1)))
    σ = stAdditive(1, Additive(ones(1)))
    LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σ])
end
let method = SemiDiscretization(0, 0.1), τmax = 1.0
    foo(a,b)::Float64 = log(spectralRadiusOfMapping(DiscreteMapping_M2(createHayesProblem(a,b), method, τmax, n_steps=10)))
    axis = [Axis(-9.0:1.0, :a), Axis(-5.0:5.0, :β)]
    pts = getinterpolatedsolution(solve!(MDBM_Problem(foo, axis), 4))
    scatter(pts..., xlim=(-9.,1.), ylim=(-5.,5.), label="",
        title="2nd moment stability border of the Hayes equation",
        xlabel=L"A", ylabel=L"$\beta$", guidefontsize=14, tickfont=font(10),
        markersize=2, markerstrokewidth=0)
    savefig(joinpath(ASSETS, "StochHayesM2.png")); println("saved StochHayesM2.png")
end

# ---- Stochastic Delay Oscillator 2nd-moment stability border ----
function createSLDOProblem(A,B,ζ,α,β,σ)
    AMx = ProportionalMX(@SMatrix [0. 1.; -A -2ζ]); τ1 = 2π
    BMx1 = DelayMX(τ1, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2); noiseID = 1
    αMx1 = stCoeffMX(noiseID, ProportionalMX(@SMatrix [0. 0.; α 0.]))
    βMx11 = stCoeffMX(noiseID, DelayMX(τ1, @SMatrix [0. 0.; β 0.]))
    σVec = stAdditive(1, Additive(@SVector [0., σ]))
    LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end
let method = SemiDiscretization(5, 2π/30), τmax = 2π+100eps()
    idxs = [1,2,3:2:(StochasticSemiDiscretizationMethod.rOfDelay(τmax,method)+1)*2...]
    foo(A,B)::Float64 = log(spectralRadiusOfMapping(DiscreteMapping_M2(
        createSLDOProblem(A,B,0.05,0.3*A,0.3*B,0.), method, τmax, idxs, n_steps=30), nev=8))
    axis = [Axis(-1.0:0.6:5.0, :A), Axis(LinRange(-1.5,1.5,12), :B)]
    pts = getinterpolatedsolution(solve!(MDBM_Problem(foo, axis), 4))
    scatter(pts..., xlim=(-1.,5.), ylim=(-1.5,1.4), label="",
        title="2nd moment stability border of the Delay Oscillator",
        xlabel=L"A", ylabel=L"$B$", guidefontsize=14, tickfont=font(10),
        markersize=2, markerstrokewidth=0)
    savefig(joinpath(ASSETS, "StochLDOM2.png")); println("saved StochLDOM2.png")
end
println("DONE readme_m2_charts")
