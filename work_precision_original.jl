using StochasticSemiDiscretizationMethod
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using StaticArrays
using Plots
using Printf
using DelimitedFiles
using Dates

# Define the Stochastic Delayed Mathieu Equation
function createStochMathieuProblem(A, ε, B, ζ, τ, σ)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε*cos(2π*t)) -2ζ]
    AMx = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2)
    noiseID = 1
    αMx1 = stCoeffMX(noiseID, ProportionalMX(@SMatrix [0. 0.; 0. 0.]))
    βMx11 = stCoeffMX(noiseID, DelayMX(τ, @SMatrix [0. 0.; 0. 0.]))
    σVec = stAdditive(1, Additive(@SVector [0., σ]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

# Parameters
A = 1.0; ε = 0.5; B = 0.2; ζ = 0.1; τ = 1.0; σ = 0.1; P = 1.0;
lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ)

# Reference value (high resolution, high order) using Multi-Free method
println("Calculating reference value using Multi-Free method...")
ref_p = 400
ref_method = SemiDiscretization(2, P/ref_p)
ref_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, ref_method, τ)
ref_dm = DiscreteMapping_M2_MF(ref_rst)
ref_rho = spectralRadiusOfMapping_MF(ref_dm)
println("Reference Spectral Radius: ", ref_rho)

orders = [0, 1, 2]
all_results = []

for order in orders
    println("\n--- Testing Order $order ---")
    
    # 30 points geometrically spaced from p=10 to ~200
    ps = unique(round.(Int, exp10.(range(log10(10), log10(200), length=30))))
    
    for p in ps
        method = SemiDiscretization(order, P/p)
        
        # Measure time
        t = @belapsed spectralRadiusOfMapping(DiscreteMapping_M2($lddep, $method, $τ, n_steps=$p))
        
        # Measure memory
        m = @allocated spectralRadiusOfMapping(DiscreteMapping_M2(lddep, method, τ, n_steps=p))
        
        # Calculate rho for error
        mapping = DiscreteMapping_M2(lddep, method, τ, n_steps=p)
        rho = spectralRadiusOfMapping(mapping)
        err = abs(rho - ref_rho)
        
        @printf("Order=%d, p=%d: time=%.4fs, mem=%.2fMB, rho=%.6f, error=%.6e\n", order, p, t, m/1024^2, rho, err)
        
        push!(all_results, [order, p, t, m, err])
        
        if t > 10.0
            println("Time limit reached for order $order at p=$p")
            break
        end
    end
end

# Save data
writedlm("baseline_results.csv", hcat(all_results...)', ',')
println("\nBaseline results saved to baseline_results.csv")

# Plotting
timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
p1 = plot(title="Work-Precision Original", xscale=:log10, yscale=:log10, ylabel="CPU Time (s)", xlabel="Resolution (p)")
p2 = plot(xscale=:log10, yscale=:log10, ylabel="Spectral Error", xlabel="Resolution (p)")
p3 = plot(xscale=:log10, yscale=:log10, ylabel="Memory (MB)", xlabel="Resolution (p)")
p4 = plot(xscale=:log10, yscale=:log10, ylabel="Spectral Error", xlabel="CPU Time (s)")

for order in orders
    res_order = filter(r -> r[1] == order, all_results)
    if isempty(res_order) continue end
    
    ps_o = [r[2] for r in res_order]
    ts_o = [r[3] for r in res_order]
    ms_o = [r[4] / 1024^2 for r in res_order]
    es_o = [r[5] for r in res_order]
    
    plot!(p1, ps_o, ts_o, marker=:o, label="Order $order")
    plot!(p2, ps_o, es_o, marker=:o, label="Order $order")
    plot!(p3, ps_o, ms_o, marker=:o, label="Order $order")
    plot!(p4, ts_o, es_o, marker=:o, label="Order $order")
end

plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
savefig("work_precision_original_$(timestamp).png")
savefig("work_precision_original.png")
