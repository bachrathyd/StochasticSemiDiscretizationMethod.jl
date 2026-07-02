# =============================================================================
# CPU vs GPU work-precision benchmark — delay-stochastic Mathieu with ALL
# matrices time-periodic (period P = 4π, delay τ = 2π):
#   A(t) = [0 1; −(A+ε cos t/2)  −2ζ]                 (parametric excitation)
#   B(t) = [0 0;  B(1+0.4 cos t/2)  0]                (periodic delayed drift)
#   α(t) = −α·A(t) rows (periodic present multiplicative noise)
#   β(t) = [0 0;  α·B(1+0.4 cos t/2)  0]              (periodic delayed noise)
#
# For each resolution p we time spectralRadiusOfMapping_MF (CPU) and
# spectralRadiusOfMapping_GPU (GPU, zero-sync), verify they agree, and plot
# the WORK-PRECISION diagram |ρ − ρ_ref| vs wall time for both.
# ρ_ref: CPU values at p = 512/1024/2048, extrapolated with the MEASURED
# convergence order (raw fine-grid SDM references are ~1e-5 biased otherwise).
#
# Run:  julia --project=. benchmark/benchmark_cpu_gpu_wp.jl
# Out:  benchmark/cpu_vs_gpu_wp.csv, benchmark/cpu_vs_gpu_wp.png
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, CUDA, Plots, Printf

CUDA.functional() || error("needs a CUDA GPU")

const Am=3.0; const εm=2.0; const Bm=0.5; const ζm=0.1; const αm=0.1
const τm=2π;  const Pm=4π

function createProblem()
    AMxfun(t) = @SMatrix [0. 1.; -(Am + εm*cos(0.5t)) -2ζm]
    BMxfun(t) = @SMatrix [0. 0.; Bm*(1+0.4cos(0.5t)) 0.]
    αMxfun(t) = @SMatrix [0. 0.; -αm*(Am + εm*cos(0.5t)) -αm*2ζm]
    βMxfun(t) = @SMatrix [0. 0.; αm*Bm*(1+0.4cos(0.5t)) 0.]
    LDDEProblem(ProportionalMX(AMxfun), [DelayMX(τm, BMxfun)],
                [stCoeffMX(1, ProportionalMX(αMxfun))],
                [stCoeffMX(1, DelayMX(τm, βMxfun))],
                Additive(2), [stAdditive(1, Additive(@SVector [0., 0.]))])
end

mapping(p; q=2) = DiscreteMapping_M2_MF(
    StochasticSemiDiscretizationMethod.calculateResults(
        createProblem(), SemiDiscretization(q, Pm/p), τm, n_steps=p))

timeit(f) = (t0=time(); v=f(); (time()-t0, v))
# best-of-3 wall time (first call already warm) — sub-10ms points are
# otherwise dominated by timer noise
function timeit3(f)
    t, v = timeit(f)
    for _ in 1:2
        t2, _ = timeit(f); t = min(t, t2)
    end
    return t, v
end

println("Warmup...")
let dm = mapping(16)
    spectralRadiusOfMapping_MF(dm); spectralRadiusOfMapping_GPU(dm)
end

# ── reference ──
# SDM q=2 converges at measured order ≈0.25 on this problem — useless as a
# reference. Use SDM q=4 (nominal O(h³)) with measured-order extrapolation,
# cross-checked against the independent v7 moment-collocation engine (GL3,
# converged to ~1e-9 by p≈32 — see highorder/).
println("Reference solves (CPU q=4, p=512/1024/2048)...")
ρr = Float64[]
for p in (512, 1024, 2048)
    t,v = timeit(() -> spectralRadiusOfMapping_MF(mapping(p; q=4)))
    push!(ρr, v); @printf("  q4 p=%4d ρ=%.10f  (%.0fs)\n", p, v, t)
end
k_meas = log2(abs(ρr[2]-ρr[1]) / abs(ρr[3]-ρr[2]))
ρ_sdm4 = ρr[3] + (ρr[3]-ρr[2]) / (2^k_meas - 1)
@printf("  SDM q4 measured order %.2f → extrapolated %.10f\n", k_meas, ρ_sdm4)

include(joinpath(@__DIR__, "..", "highorder", "cov_colloc_v7.jl"))
pb_v7 = Prob(2, Pm, τm,
    t->[0.0 1.0; -(Am + εm*cos(0.5t)) -2ζm],
    t->[0.0 0.0; Bm*(1+0.4cos(0.5t)) 0.0],
    t->[0.0 0.0; -αm*(Am + εm*cos(0.5t)) -αm*2ζm],
    t->[0.0 0.0; αm*Bm*(1+0.4cos(0.5t)) 0.0])
ρv7a = rho_H_krylov(build_v7(pb_v7, 3, 32))
ρv7b = rho_H_krylov(build_v7(pb_v7, 3, 48))
@printf("  v7 GL3: p=32 %.10f  p=48 %.10f  (Δ=%.1e)\n", ρv7a, ρv7b, abs(ρv7a-ρv7b))
@printf("  |v7 − SDMq4-extrap| = %.2e\n", abs(ρv7b-ρ_sdm4))
ρ_ref = abs(ρv7a-ρv7b) < 1e-7 ? ρv7b : ρ_sdm4
@printf("ρ_ref = %.10f\n", ρ_ref)

# ── sweep ──
ps = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
rows = NamedTuple[]
for p in ps
    dm = mapping(p)
    t_cpu, ρ_cpu = timeit3(() -> spectralRadiusOfMapping_MF(dm))
    t_gpu, ρ_gpu = timeit3(() -> spectralRadiusOfMapping_GPU(dm))
    err  = abs(ρ_cpu - ρ_ref)
    dis  = abs(ρ_cpu - ρ_gpu)/max(ρ_cpu,1e-300)
    push!(rows, (p=p, t_cpu=t_cpu, t_gpu=t_gpu, ρ_cpu=ρ_cpu, ρ_gpu=ρ_gpu, err=err, dis=dis))
    @printf("p=%5d  CPU %8.3fs  GPU %8.3fs  ρ=%.10f  err=%.2e  cpu/gpu-mismatch=%.1e %s\n",
            p, t_cpu, t_gpu, ρ_cpu, err, dis, dis < 1e-8 ? "OK" : "MISMATCH!")
    flush(stdout)
end

open(joinpath(@__DIR__,"cpu_vs_gpu_wp.csv"),"w") do io
    println(io,"p,t_cpu,t_gpu,rho_cpu,rho_gpu,err")
    for r in rows
        @printf(io,"%d,%.6f,%.6f,%.12f,%.12f,%.6e\n",r.p,r.t_cpu,r.t_gpu,r.ρ_cpu,r.ρ_gpu,r.err)
    end
end

errs =[max(r.err,1e-13) for r in rows]
tcpu =[max(r.t_cpu,1e-4) for r in rows]
tgpu =[max(r.t_gpu,1e-4) for r in rows]
p1 = plot(tcpu, errs, marker=:circle, label="CPU (MF)",
          xscale=:log10, yscale=:log10,
          xlabel="wall time [s]", ylabel="|ρ(H) − ρ_ref|",
          title="Work-precision: CPU vs GPU\nfully periodic delay-stoch. Mathieu (d=2, q=2)",
          legend=:topright)
plot!(p1, tgpu, errs, marker=:star5, label="GPU (zero-sync)")
p2 = plot([r.p for r in rows], tcpu, marker=:circle, label="CPU (MF)",
          xscale=:log10, yscale=:log10, xlabel="p (steps / period)", ylabel="wall time [s]",
          title="Time vs resolution", legend=:topleft)
plot!(p2, [r.p for r in rows], tgpu, marker=:star5, label="GPU (zero-sync)")
savefig(plot(p1, p2, layout=(1,2), size=(1400,560)),
        joinpath(@__DIR__,"cpu_vs_gpu_wp.png"))
println("wrote benchmark/cpu_vs_gpu_wp.csv and benchmark/cpu_vs_gpu_wp.png")
