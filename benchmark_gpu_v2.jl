using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra
using StaticArrays
using BenchmarkTools
using Printf
using Dates
using CUDA

BLAS.set_num_threads(1)

# Stochastic Mathieu (d=2) — same problem as the CPU complexity benchmark
function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val, P)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε*cos(2π*t/P)) -2ζ]
    AMx   = ProportionalMX(AMxfun)
    BMx1  = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec  = Additive(2)
    αMx1  = stCoeffMX(1, ProportionalMX(@SMatrix [0. 0.; α_val 0.]))
    βMx11 = stCoeffMX(1, DelayMX(τ, @SMatrix [0. 0.; 0. 0.]))
    σVec  = stAdditive(1, Additive(@SVector [0., σ]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

A=1.0; ε=0.5; B=0.2; ζ=0.1; τ=1.0; σ=0.1; α_val=0.2; P=1.0
lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val, P)

ps    = [20, 100, 500]
order = 1

println("GPU device: ", CUDA.name(CUDA.device()))
println("Benchmark: SemiDiscretization order=$order, d=2 Mathieu\n")

ts_stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
csv_path = "benchmark_gpu_v2_$(ts_stamp).csv"

open(csv_path, "w") do io
    println(io, "p,D,rho_cpu,rho_gpu_v1,rho_gpu_v2,rho_gpu_v3,t_cpu_s,t_gpu_v1_s,t_gpu_v2_s,t_gpu_v3_s,speedup_v1,speedup_v2,speedup_v3")
end

for p in ps
    method = SemiDiscretization(order, P/p)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)

    d_sys = 2
    r     = div(rst.n, d_sys) - 1
    D     = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d_sys).sectionStarts[end]

    @printf("\n=== p=%d  r=%d  D=%d ===\n", p, r, D)

    # --- Warm-up (forces JIT + CUDA compilation) ---
    print("  Warming up... ")
    ρ_cpu    = spectralRadiusOfMapping_MF(dm)
    ρ_gpu_v1 = spectralRadiusOfMapping_GPU(dm)
    ρ_gpu_v2 = spectralRadiusOfMapping_GPU_v2(dm)
    ρ_gpu_v3 = spectralRadiusOfMapping_GPU_v3(dm)
    CUDA.synchronize()
    @printf("ρ_cpu=%.8f  ρ_v1=%.8f  ρ_v2=%.8f  ρ_v3=%.8f\n", ρ_cpu, ρ_gpu_v1, ρ_gpu_v2, ρ_gpu_v3)

    # Check agreement
    err_v1 = abs(ρ_gpu_v1 - ρ_cpu) / abs(ρ_cpu)
    err_v2 = abs(ρ_gpu_v2 - ρ_cpu) / abs(ρ_cpu)
    err_v3 = abs(ρ_gpu_v3 - ρ_cpu) / abs(ρ_cpu)
    @printf("  Relative error  v1: %.2e   v2: %.2e   v3: %.2e\n", err_v1, err_v2, err_v3)
    if err_v3 > 1e-6
        println("  WARNING: v3 result differs from CPU by more than 1e-6!")
    end

    # --- Timed runs ---
    print("  Timing CPU...     ")
    bm_cpu = @benchmark spectralRadiusOfMapping_MF($dm) samples=3 evals=1 seconds=30
    t_cpu  = median(bm_cpu).time / 1e9
    @printf("%.4f s\n", t_cpu)

    print("  Timing GPU v1...  ")
    bm_v1 = @benchmark (spectralRadiusOfMapping_GPU($dm); CUDA.synchronize()) samples=3 evals=1 seconds=30
    t_v1  = median(bm_v1).time / 1e9
    @printf("%.4f s\n", t_v1)

    print("  Timing GPU v2...  ")
    bm_v2 = @benchmark (spectralRadiusOfMapping_GPU_v2($dm); CUDA.synchronize()) samples=3 evals=1 seconds=30
    t_v2  = median(bm_v2).time / 1e9
    @printf("%.4f s\n", t_v2)

    print("  Timing GPU v3...  ")
    bm_v3 = @benchmark (spectralRadiusOfMapping_GPU_v3($dm); CUDA.synchronize()) samples=3 evals=1 seconds=30
    t_v3  = median(bm_v3).time / 1e9
    @printf("%.4f s\n", t_v3)

    @printf("  Speedup v1 vs CPU: %.2fx\n", t_cpu/t_v1)
    @printf("  Speedup v2 vs CPU: %.2fx\n", t_cpu/t_v2)
    @printf("  Speedup v3 vs CPU: %.2fx\n", t_cpu/t_v3)
    @printf("  Speedup v3 vs v2:  %.2fx\n", t_v2/t_v3)

    open(csv_path, "a") do io
        @printf(io, "%d,%d,%.10f,%.10f,%.10f,%.10f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                p, D, ρ_cpu, ρ_gpu_v1, ρ_gpu_v2, ρ_gpu_v3,
                t_cpu, t_v1, t_v2, t_v3, t_cpu/t_v1, t_cpu/t_v2, t_cpu/t_v3)
    end
end

println("\nDone. Results → $csv_path")
