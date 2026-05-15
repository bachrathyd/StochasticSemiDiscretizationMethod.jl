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
function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε*cos(2π*t)) -2ζ]
    AMx = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2)
    noiseID = 1
    αMx1 = stCoeffMX(noiseID, ProportionalMX(@SMatrix [0. 0.; α_val 0.]))
    βMx11 = stCoeffMX(noiseID, DelayMX(τ, @SMatrix [0. 0.; 0. 0.]))
    σVec = stAdditive(1, Additive(@SVector [0., σ]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

# Parameters
A = 1.0; ε = 0.5; B = 0.2; ζ = 0.1; τ = 1.0; σ = 0.1; α_val = 0.2; P = 1.0;
lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)

# Reference value (high resolution, high order)
println("Calculating reference value...")
ref_p = 300
ref_method = SemiDiscretization(2, P/ref_p)
ref_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, ref_method, τ)
ref_dm = DiscreteMapping_M2_MF(ref_rst)
ref_rho = spectralRadiusOfMapping_MF(ref_dm)
println("Reference Spectral Radius: ", ref_rho)

# Load baseline data
baseline_results = nothing
if isfile("baseline_results.csv")
    baseline_results = readdlm("baseline_results.csv", ',')
end

orders = [0, 1, 2]
all_mf_results = []

for order in orders
    println("\n--- Testing Order $order (Multi-Free) ---")
    
    # 15 points geometrically spaced from p=10 to ~1000
    ps = unique(round.(Int, exp10.(range(log10(10), log10(1000), length=15))))
    
    for p in ps
        method = SemiDiscretization(order, P/p)
        rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
        dm = DiscreteMapping_M2_MF(rst)
        
        # Measure time
        t = @belapsed spectralRadiusOfMapping_MF($dm)
        
        # Measure memory
        m = @allocated spectralRadiusOfMapping_MF(dm)
        
        # Spectral radius for error
        rho = spectralRadiusOfMapping_MF(dm)
        err = abs(rho - ref_rho)
        
        @printf("Order=%d, p=%d: time=%.4fs, mem=%.2fMB, rho=%.6f, error=%.6e\n", order, p, t, m/1024^2, rho, err)
        
        push!(all_mf_results, [order, p, t, m, err])
        
        if t > 10.0
            println("Time limit reached for order $order at p=$p")
            break
        end
    end
end

# Plotting
timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
p1 = plot(title="Work-Precision Comparison", xscale=:log10, yscale=:log10, ylabel="CPU Time (s)", xlabel="Resolution (p)")
p2 = plot(xscale=:log10, yscale=:log10, ylabel="Spectral Error", xlabel="Resolution (p)")
p3 = plot(xscale=:log10, yscale=:log10, ylabel="Memory (MB)", xlabel="Resolution (p)")
p4 = plot(xscale=:log10, yscale=:log10, ylabel="Spectral Error", xlabel="CPU Time (s)")

for order in orders
    # MF results
    res_mf = filter(r -> r[1] == order, all_mf_results)
    if !isempty(res_mf)
        ps_m = [r[2] for r in res_mf]
        ts_m = [r[3] for r in res_mf]
        ms_m = [r[4] / 1024^2 for r in res_mf]
        es_m = [r[5] for r in res_mf]
        
        plot!(p1, ps_m, ts_m, marker=:o, label="MF Order $order")
        plot!(p2, ps_m, es_m, marker=:o, label="MF Order $order")
        plot!(p3, ps_m, ms_m, marker=:o, label="MF Order $order")
        plot!(p4, ts_m, es_m, marker=:o, label="MF Order $order")
    end
    
    # Baseline results
    if !isnothing(baseline_results)
        res_orig = baseline_results[baseline_results[:,1] .== order, :]
        if !isempty(res_orig)
            plot!(p1, res_orig[:,2], res_orig[:,3], marker=:s, linestyle=:dash, label="Orig Order $order")
            plot!(p2, res_orig[:,2], res_orig[:,5], marker=:s, linestyle=:dash, label="Orig Order $order")
            plot!(p3, res_orig[:,2], res_orig[:,4] ./ 1024^2, marker=:s, linestyle=:dash, label="Orig Order $order")
            plot!(p4, res_orig[:,3], res_orig[:,5], marker=:s, linestyle=:dash, label="Orig Order $order")
        end
    end
end

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig("work_precision_multifree_vs_original_$(timestamp).png")
savefig("work_precision_multifree.png")

println("\nComparison diagram saved.")
