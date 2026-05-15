using Pkg
Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra
using BenchmarkTools
using Dates

function createHayesProblem(a, β)
    AMx  = ProportionalMX(a*ones(1,1))
    τ1   = 1.0
    BMx1 = DelayMX(τ1, zeros(1,1))
    cVec = Additive(1)
    noiseID = 1
    αMx1  = stCoeffMX(noiseID, ProportionalMX(zeros(1,1)))
    βMx11 = stCoeffMX(noiseID, DelayMX(τ1, β*ones(1,1)))
    σ     = stAdditive(1, Additive(ones(1)))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σ])
end

a = -6.0; β = 2.0; τ = 1.0
ps = [20, 100, 500]

results = Vector{NamedTuple{(:p, :rho, :time_s, :date), Tuple{Int, Float64, Float64, String}}}()

for p_target in ps
    Δt  = τ / p_target
    lddep  = createHayesProblem(a, β)
    method = SemiDiscretization(0, Δt)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)

    println("=== p = $p_target (actual steps = $(rst.n_steps)) ===")

    # warm-up
    ρ = spectralRadiusOfMapping_MF(dm)
    println("  ρ = $ρ  (warm-up)")

    # timed run (at least 3 samples, 10 s budget)
    t = @belapsed spectralRadiusOfMapping_MF($dm) seconds=10 samples=3
    println("  CPU time: $t s")

    push!(results, (p=rst.n_steps, rho=ρ, time_s=t, date=string(now())))
end

println("\n--- Summary ---")
csv_path = joinpath(@__DIR__, "baseline_cpu_performance.csv")
open(csv_path, "w") do io
    println(io, "p,rho,time_s,date")
    for r in results
        println(io, "$(r.p),$(r.rho),$(r.time_s),$(r.date)")
        println("  p=$(r.p)  ρ=$(r.rho)  t=$(r.time_s) s  [$(r.date)]")
    end
end
println("\nSaved → $csv_path")
