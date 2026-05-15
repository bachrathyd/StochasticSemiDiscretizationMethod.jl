using StochasticSemiDiscretizationMethod
using BenchmarkTools
using StaticArrays
using LinearAlgebra
using Plots
using Printf
using Statistics

# Define the Stochastic Delayed Mathieu Equation
function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)
    # Parametric excitation period is 2π
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε*cos(t)) -2ζ]
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
A = 1.0; ε = 0.5; B = 0.2; ζ = 0.1; τ = 2π; σ = 0.1; α_val = 0.2; P = 2π;
lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)

# Reference value (high resolution, high order)
println("Calculating reference value...")
ref_p = 400
ref_method = SemiDiscretization(2, P/ref_p)
ref_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, ref_method, τ)
ref_dm = DiscreteMapping_M2_MF(ref_rst)
ref_rho = spectralRadiusOfMapping_MF(ref_dm)
println("Reference Spectral Radius: ", ref_rho)

orders = [0, 1, 2, 3]
# Resolutions from 5 to 100,000
ps = unique(round.(Int, exp10.(range(log10(5), log10(100000), length=30))))

# results[(order, method_type, p)] = (time, mem, rho, err)
# method_type: 1 = Original, 2 = MF
results = Dict{Tuple{Int, Int, Int}, Tuple{Float64, Float64, Float64, Float64}}()

stop_orig = Dict(ord => false for ord in orders)
stop_mf = Dict(ord => false for ord in orders)

function fit_complexity(x, y)
    valid = (x .> 0) .& (y .> 1e-15)
    if sum(valid) < 3 return NaN end
    lx = log10.(x[valid])
    ly = log10.(y[valid])
    
    # Filter points in noise floor
    valid_noise = ly .> -14.5
    lx = lx[valid_noise]
    ly = ly[valid_noise]
    
    if length(lx) < 3 return NaN end
    
    # Remove first 20% of points (small points)
    n_skip = max(1, length(lx) ÷ 5)
    lx = lx[n_skip:end]
    ly = ly[n_skip:end]
    
    if length(lx) < 2 return NaN end
    
    slopes = [(ly[i+1] - ly[i]) / (lx[i+1] - lx[i]) for i in 1:length(lx)-1]
    return median(slopes)
end

for p in ps
    println("\n--- Testing p = $p ---")
    for order in orders
        method = SemiDiscretization(order, P/p)
        
        # MF version
        if !stop_mf[order]
            print("  MF Order $order: ")
            try
                rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
                dm = DiscreteMapping_M2_MF(rst)
                
                # Use @benchmark for better accuracy
                bm = @benchmark spectralRadiusOfMapping_MF($dm) samples=1 evals=1 seconds=5
                t = median(bm).time / 1e9
                m = bm.memory
                rho = spectralRadiusOfMapping_MF(dm)
                err = abs(rho - ref_rho)
                
                results[(order, 2, p)] = (t, m, rho, err)
                @printf("time=%.4fs, err=%.2e\n", t, err)
                if t > 5.0 
                    println("    Time limit reached for MF Order $order")
                    stop_mf[order] = true 
                end
            catch e
                println("    MF failed: ", e)
                stop_mf[order] = true
            end
        end

        # Original version
        if !stop_orig[order]
            print("  Original Order $order: ")
            try
                # We limit original method to smaller p to avoid hang
                if p > 1000 && order > 0
                    println("    Skipping Original for high p")
                    stop_orig[order] = true
                else
                    rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
                    dm = DiscreteMapping_M2(rst)
                    
                    bm = @benchmark spectralRadiusOfMapping($dm) samples=1 evals=1 seconds=5
                    t = median(bm).time / 1e9
                    m = bm.memory
                    rho = spectralRadiusOfMapping(dm)
                    err = abs(rho - ref_rho)
                    
                    results[(order, 1, p)] = (t, m, rho, err)
                    @printf("time=%.4fs, err=%.2e\n", t, err)
                    if t > 5.0 
                        println("    Time limit reached for Original Order $order")
                        stop_orig[order] = true 
                    end
                end
            catch e
                println("    Original failed: ", e)
                stop_orig[order] = true
            end
        end
    end
    
    # Update plots in every step of the outer loop
    p1 = plot(title="Time vs Resolution", xlabel="p", ylabel="Time (s)", xscale=:log10, yscale=:log10, grid=:both, minorgrid=true)
    p2 = plot(title="Error vs Resolution", xlabel="p", ylabel="Error", xscale=:log10, yscale=:log10, grid=:both, minorgrid=true)
    p3 = plot(title="Memory vs Resolution", xlabel="p", ylabel="Memory (MB)", xscale=:log10, yscale=:log10, grid=:both, minorgrid=true)
    p4 = plot(title="Error vs Time", xlabel="Time (s)", ylabel="Error", xscale=:log10, yscale=:log10, grid=:both, minorgrid=true)

    for ord in orders
        for (m_type, label_prefix, marker) in [(1, "Orig", :circle), (2, "MF", :rect)]
            # Filter results for this order and method
            ps_ord = sort([k[3] for k in keys(results) if k[1] == ord && k[2] == m_type])
            if isempty(ps_ord) continue end
            
            res_list = [results[(ord, m_type, p_val)] for p_val in ps_ord]
            ts = [r[1] for r in res_list]
            ms = [r[2]/1024^2 for r in res_list]
            es = [r[4] for r in res_list]
            
            c_time = fit_complexity(ps_ord, ts)
            c_err = fit_complexity(ps_ord, es)
            
            label = @sprintf("%s Ord %d (T_slope:%.1f, E_slope:%.1f)", label_prefix, ord, c_time, c_err)
            
            plot!(p1, ps_ord, ts, marker=marker, label=label)
            plot!(p2, ps_ord, es, marker=marker, label=label)
            plot!(p3, ps_ord, ms, marker=marker, label=label)
            plot!(p4, ts, es, marker=marker, label=label)
        end
    end
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
    savefig(final_plot, "work_precision_benchmark.png")
    
    # If all methods stopped, break
    if all(stop_orig[ord] for ord in orders) && all(stop_mf[ord] for ord in orders)
        println("\nFinal Complexity Summary:")
        println("-"^80)
        @printf("%-15s %-5s | %-15s | %-15s\n", "Method", "Ord", "Time Complexity", "Error Complexity")
        println("-"^80)
        for ord in orders
            for (m_type, label_prefix) in [(1, "Original"), (2, "MF")]
                ps_ord = sort([k[3] for k in keys(results) if k[1] == ord && k[2] == m_type])
                if isempty(ps_ord) continue end
                res_list = [results[(ord, m_type, p_val)] for p_val in ps_ord]
                ts = [r[1] for r in res_list]
                es = [r[4] for r in res_list]
                c_time = fit_complexity(ps_ord, ts)
                c_err = fit_complexity(ps_ord, es)
                @printf("%-15s %-5d | %-15.2f | %-15.2f\n", label_prefix, ord, c_time, c_err)
            end
        end
        println("-"^80)
        println("All methods reached time limit.")
        break
    end
end

println("\nBenchmark complete. Plot saved as work_precision_benchmark.png")
