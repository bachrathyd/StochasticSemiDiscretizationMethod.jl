# CPU (multiplication-free) vs GPU spectral-radius benchmark
#
# Problem: stochastic delayed Mathieu equation (d = 2, q = 2, period 4π, τ = 2π),
# the Sykora-2020 benchmark. For each period resolution p we time
#   spectralRadiusOfMapping_MF   (CPU, multiplication-free Krylov)
#   spectralRadiusOfMapping_GPU  (GPU, zero-sync device-resident Krylov)
# and verify both return the same ρ(H).
#
# Run:  julia --project=. benchmark/benchmark_cpu_gpu.jl
# Output: benchmark/cpu_vs_gpu.csv, benchmark/cpu_vs_gpu.png
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
using StaticArrays
using CUDA
using Plots
using Printf

CUDA.functional() || error("This benchmark requires a functional CUDA GPU.")

function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε * cos(0.5 * t)) -2ζ]
    AMx  = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2)
    αMxfun(t) = @SMatrix [0. 0.; -α_val*(A + ε*cos(0.5*t)) -α_val*2ζ]
    αMx1  = stCoeffMX(1, ProportionalMX(αMxfun))
    βMx11 = stCoeffMX(1, DelayMX(τ, @SMatrix [0. 0.; α_val*B 0.]))
    σVec  = stAdditive(1, Additive(@SVector [0., σ]))
    LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

const τ = 2π
const P = 4π
lddep = createStochMathieuProblem(3.0, 2.0, 0.5, 0.1, τ, 0.0, 0.1)

mapping(p) = DiscreteMapping_M2_MF(
    StochasticSemiDiscretizationMethod.calculateResults(
        lddep, SemiDiscretization(2, P/p), τ, n_steps=p))

println("Warmup (JIT)...")
let dm = mapping(10)
    spectralRadiusOfMapping_MF(dm)
    spectralRadiusOfMapping_GPU(dm)
end

ps       = [10, 20, 40, 70, 100, 150, 220, 320, 460, 640, 900, 1280]
TIME_CAP = 600.0   # s — a method is dropped once it exceeds this

timeit(f) = (t0 = time(); v = f(); (time() - t0, v))

rows = NamedTuple[]
stop_cpu = false; stop_gpu = false
for p in ps
    dm = mapping(p)
    d = 2; r = div(dm.rst.n, d) - 1
    D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]

    t_cpu = NaN; ρ_cpu = NaN
    if !stop_cpu
        t_cpu, ρ_cpu = timeit(() -> spectralRadiusOfMapping_MF(dm))
        t_cpu > TIME_CAP && (global stop_cpu = true)
    end
    t_gpu = NaN; ρ_gpu = NaN
    if !stop_gpu
        t_gpu, ρ_gpu = timeit(() -> spectralRadiusOfMapping_GPU(dm))
        t_gpu > TIME_CAP && (global stop_gpu = true)
    end

    push!(rows, (p=p, D=D, t_cpu=t_cpu, t_gpu=t_gpu, ρ_cpu=ρ_cpu, ρ_gpu=ρ_gpu))
    @printf("p=%5d D=%8d  CPU %9.3fs  GPU %9.3fs  speedup %5.2f×  ρ agree: %s\n",
            p, D, t_cpu, t_gpu, t_cpu/t_gpu,
            (isnan(ρ_cpu) || isnan(ρ_gpu)) ? "-" :
                (abs(ρ_cpu-ρ_gpu)/abs(ρ_cpu) < 1e-8 ? "yes" : "NO!"))
    flush(stdout)
    (stop_cpu && stop_gpu) && break
end

open(joinpath(@__DIR__, "cpu_vs_gpu.csv"), "w") do io
    println(io, "p,D,t_cpu,t_gpu,rho_cpu,rho_gpu")
    for r in rows
        println(io, "$(r.p),$(r.D),$(r.t_cpu),$(r.t_gpu),$(r.ρ_cpu),$(r.ρ_gpu)")
    end
end

Ds     = [r.D for r in rows]
t_cpus = [r.t_cpu for r in rows]
t_gpus = [r.t_gpu for r in rows]
plt = plot(Ds, t_cpus, marker=:o, label="CPU (MF)",
           xscale=:log10, yscale=:log10,
           xlabel="second-moment state size D", ylabel="wall time [s]",
           title="ρ(H): CPU vs GPU — stochastic delay Mathieu (d=2, q=2)",
           legend=:topleft)
plot!(plt, Ds, t_gpus, marker=:s, label="GPU (zero-sync)")
savefig(plt, joinpath(@__DIR__, "cpu_vs_gpu.png"))
println("Wrote benchmark/cpu_vs_gpu.csv and benchmark/cpu_vs_gpu.png")
