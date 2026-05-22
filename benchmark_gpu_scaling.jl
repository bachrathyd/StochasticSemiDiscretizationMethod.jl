using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra, StaticArrays, CUDA, Printf, Dates, Plots

BLAS.set_num_threads(1)

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
order  = 1
ρ_ref  = 0.6817666221   # high-res reference (order=2, p=500)

TIME_LIMIT = 10.0       # s — stop adding points after this threshold

println("GPU : ", CUDA.name(CUDA.device()))
@printf("VRAM: %.2f GB total,  %.2f GB free\n\n",
        CUDA.total_memory()/1024^3, CUDA.available_memory()/1024^3)

# ---- Pre-warm: trigger JIT for both GPU and CPU functions ----
print("Pre-warming GPU kernel (JIT compilation) ... ")
_prewarm_method = SemiDiscretization(order, P/20)
_prewarm_rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, _prewarm_method, τ)
_prewarm_dm     = DiscreteMapping_M2_MF(_prewarm_rst)
t_jit = @elapsed begin
    spectralRadiusOfMapping_GPU_v3(_prewarm_dm)
    CUDA.synchronize()
end
@printf("done (%.1f s JIT)\n", t_jit)
print("Pre-warming CPU function ... ")
spectralRadiusOfMapping_MF(_prewarm_dm)   # trigger CPU JIT
println("done")
println()
_prewarm_dm = nothing; _prewarm_rst = nothing; _prewarm_method = nothing; GC.gc()

# p values to benchmark
ps_all = [10, 25, 50, 100, 225, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000]
ps_cpu = [10, 25, 50, 100, 225, 500]   # actually run CPU here; extrapolate beyond

ts       = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
csv_path = "benchmark_gpu_scaling_$(ts).csv"
png_path = "benchmark_gpu_scaling_$(ts).png"

open(csv_path, "w") do io
    println(io, "p,r,D,C_MB,alloc_GB,rho_gpu,err_gpu,t_setup_s,t_cpu_s,t_gpu_s,speedup")
end

struct Row
    p::Int; r::Int; D::Int
    C_MB::Float64; alloc_GB::Float64
    rho_gpu::Float64; err_gpu::Float64
    t_setup::Float64; t_cpu::Float64; t_gpu::Float64
end
rows = Row[]

for p in ps_all
    # ---- CPU-side setup (build coefficients) ----
    method  = SemiDiscretization(order, P/p)
    t_setup = @elapsed rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm      = DiscreteMapping_M2_MF(rst)

    d_s = 2
    r   = div(rst.n, d_s) - 1
    D   = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d_s).sectionStarts[end]

    C_bytes    = Float64((r+1)^2) * d_s^2 * 8.0
    D_bytes    = Float64(D) * 8.0
    alloc_bytes = C_bytes + 2.0*D_bytes   # C + two state vectors (approx)
    C_MB       = C_bytes  / 1024^2
    alloc_GB   = alloc_bytes / 1024^3

    @printf("\n=== p=%5d  r=%5d  D=%10d  C=%7.1f MB  alloc≈%.2f GB ===\n",
            p, r, D, C_MB, alloc_GB)
    @printf("  Setup: %.2f s   Free GPU: %.2f GB\n",
            t_setup, CUDA.available_memory()/1024^3)

    # ---- Memory check (C matrix + 2 state vectors + Krylov basis at kd=30) ----
    krylov_bytes = 30.0 * D_bytes          # kd=30 Krylov vectors
    total_needed = alloc_bytes + krylov_bytes
    avail_now    = Float64(CUDA.available_memory())
    if total_needed > avail_now * 0.85
        @printf("  SKIP — need %.2f GB (incl. Krylov basis) but only %.2f GB free\n",
                total_needed/1024^3, avail_now/1024^3)
        break
    end

    # ---- GPU single-shot timing (JIT already done by pre-warm) ----
    ρ_gpu = NaN;  t_gpu = NaN
    try
        print("  GPU ... ")
        t_gpu = @elapsed begin
            ρ_gpu = spectralRadiusOfMapping_GPU_v3(dm)
            CUDA.synchronize()
        end
        err = abs(ρ_gpu - ρ_ref) / ρ_ref
        @printf("ρ=%.8f  err=%.2e  t=%.4f s\n", ρ_gpu, err, t_gpu)
    catch e
        estr = string(e)
        if contains(estr, "out of memory") || contains(estr, "OutOfMemory")
            println("  → GPU OUT OF MEMORY")
        else
            println("  → ERROR: $e")
        end
        break
    end

    # ---- CPU single-shot timing (small p only) ----
    t_cpu = NaN
    if p in ps_cpu
        print("  CPU ... ")
        t_cpu = @elapsed spectralRadiusOfMapping_MF(dm)
        @printf("%.4f s   speedup=%.2fx\n", t_cpu, t_cpu/t_gpu)
    end

    # Free GPU memory before next iteration to avoid Krylov-basis OOM at large p
    GC.gc(); CUDA.reclaim()

    err_gpu = abs(ρ_gpu - ρ_ref) / ρ_ref
    push!(rows, Row(p, r, D, C_MB, alloc_GB, ρ_gpu, err_gpu, t_setup, t_cpu, t_gpu))

    open(csv_path, "a") do io
        @printf(io, "%d,%d,%d,%.2f,%.4f,%.10f,%.2e,%.4f,%.6f,%.6f,%.4f\n",
                p, r, D, C_MB, alloc_GB, ρ_gpu, err_gpu, t_setup,
                isnan(t_cpu) ? -1.0 : t_cpu,
                isnan(t_gpu) ? -1.0 : t_gpu,
                (isnan(t_cpu)||isnan(t_gpu)) ? -1.0 : t_cpu/t_gpu)
    end

    # ---- live plot ----
    valid = filter(rw -> !isnan(rw.t_gpu), rows)
    if length(valid) >= 2
        pv  = [rw.p       for rw in valid]
        tv  = [rw.t_gpu   for rw in valid]
        Dv  = [rw.D       for rw in valid]
        ev  = [rw.err_gpu for rw in valid]
        CMv = [rw.C_MB    for rw in valid]

        cpu_rows = filter(rw -> !isnan(rw.t_cpu), valid)
        pv_cpu   = [rw.p     for rw in cpu_rows]
        tv_cpu   = [rw.t_cpu for rw in cpu_rows]

        # Power-law fit for CPU extrapolation: t_cpu = A * p^b
        cpu_extrap_p = Float64[]; cpu_extrap_t = Float64[]
        if length(cpu_rows) >= 2
            lp = log10.(Float64.(pv_cpu))
            lt = log10.(tv_cpu)
            n_fit = length(lp)
            b_fit = (n_fit * sum(lp .* lt) - sum(lp)*sum(lt)) / (n_fit*sum(lp.^2) - sum(lp)^2)
            a_fit = (sum(lt) - b_fit*sum(lp)) / n_fit
            p_extrap_range = range(Float64(pv[1]), Float64(pv[end]), length=200)
            cpu_extrap_p = collect(p_extrap_range)
            cpu_extrap_t = 10 .^ (a_fit .+ b_fit .* log10.(cpu_extrap_p))
        end

        # slope helper (log-log slope of last half of data)
        function slope(xs, ys)
            lx = log10.(Float64.(xs)); ly = log10.(ys)
            length(lx) < 3 && return NaN
            n = length(lx); i0 = max(1, n÷2)
            slopes = diff(ly[i0:end]) ./ diff(lx[i0:end])
            round(sum(slopes)/length(slopes), digits=2)
        end

        s_t = slope(pv, tv); s_e = slope(pv, ev); s_D = slope(pv, Dv)

        p1 = plot(pv, tv, marker=:circle, color=:blue, lw=2,
                  label="GPU v3 (slope≈$s_t)",
                  title="Compute time vs p", xlabel="p", ylabel="time (s)",
                  xscale=:log10, yscale=:log10, legend=:topleft)
        if !isempty(pv_cpu)
            plot!(p1, pv_cpu, tv_cpu, marker=:square, color=:red, lw=2, label="CPU MF (measured)")
        end
        if !isempty(cpu_extrap_p)
            plot!(p1, cpu_extrap_p, cpu_extrap_t, ls=:dash, color=:red, lw=1, label="CPU MF (extrap.)")
        end
        hline!(p1, [TIME_LIMIT], ls=:dash, color=:gray, lw=1, label="10 s limit")
        if length(pv) >= 2
            p0f = Float64(pv[1]); t0f = tv[1]
            pr = [Float64(pv[1]), Float64(pv[end])]
            plot!(p1, pr, t0f .* (pr./p0f).^2, ls=:dot, color=:lightgray, lw=1, label="O(p²)")
            plot!(p1, pr, t0f .* (pr./p0f).^3, ls=:dot, color=:gray,      lw=1, label="O(p³)")
        end

        p2 = plot(pv, ev, marker=:circle, color=:blue, lw=2,
                  label="GPU v3 (slope≈$s_e)",
                  title="Error vs p (ref ρ_ref=0.68177)",
                  xlabel="p", ylabel="|ρ - ρ_ref| / ρ_ref",
                  xscale=:log10, yscale=:log10, legend=:topright)

        p3 = plot(pv, Dv, marker=:circle, color=:green, lw=2,
                  label="D (slope≈$s_D)",
                  title="State-space dim D vs p", xlabel="p", ylabel="D",
                  xscale=:log10, yscale=:log10, legend=:topleft)

        p4 = plot(pv, CMv, marker=:diamond, color=:purple, lw=2,
                  label="C array",
                  title="GPU memory usage vs p", xlabel="p", ylabel="MB",
                  xscale=:log10, yscale=:log10, legend=:topleft)

        fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1300,1000),
                   plot_title="GPU v3 Scaling — Stochastic Mathieu d=2, order=$order")
        savefig(fig, png_path)
    end

    if !isnan(t_gpu) && t_gpu > TIME_LIMIT
        @printf("\n→ 10 s limit reached at p=%d  (t=%.2f s)\n", p, t_gpu)
        break
    end
end

# ---- CPU power-law fit summary ----
cpu_rows_all = filter(rw -> !isnan(rw.t_cpu), rows)
if length(cpu_rows_all) >= 2
    pv_c = Float64.([rw.p for rw in cpu_rows_all])
    tv_c = [rw.t_cpu for rw in cpu_rows_all]
    lp = log10.(pv_c); lt = log10.(tv_c)
    n_fit = length(lp)
    b_fit = (n_fit*sum(lp.*lt) - sum(lp)*sum(lt)) / (n_fit*sum(lp.^2) - sum(lp)^2)
    a_fit = (sum(lt) - b_fit*sum(lp)) / n_fit
    println("\nCPU power-law fit: t_cpu ≈ 10^$(round(a_fit,digits=4)) × p^$(round(b_fit,digits=3)) s")
end

println("\n\nSummary:")
println("-"^90)
@printf("%-6s  %-10s  %-10s  %-12s  %-14s  %-14s  %s\n",
        "p", "D", "C(MB)", "t_cpu(s)", "t_cpu_extrap(s)", "t_gpu(s)", "err_vs_ref")
for rw in rows
    t_cpu_str = if !isnan(rw.t_cpu)
        @sprintf("%-12.4f", rw.t_cpu)
    elseif length(cpu_rows_all) >= 2
        pv_c = Float64.([r.p for r in cpu_rows_all])
        tv_c = [r.t_cpu for r in cpu_rows_all]
        lp = log10.(pv_c); lt = log10.(tv_c)
        n_fit = length(lp)
        b_fit = (n_fit*sum(lp.*lt) - sum(lp)*sum(lt)) / (n_fit*sum(lp.^2) - sum(lp)^2)
        a_fit = (sum(lt) - b_fit*sum(lp)) / n_fit
        t_extrap = 10^(a_fit + b_fit*log10(Float64(rw.p)))
        @sprintf("~%-11.1f", t_extrap)
    else
        @sprintf("%-12s", "—")
    end
    spd_str = if !isnan(rw.t_cpu)
        @sprintf("%.2fx", rw.t_cpu/rw.t_gpu)
    elseif length(cpu_rows_all) >= 2
        pv_c = Float64.([r.p for r in cpu_rows_all])
        tv_c = [r.t_cpu for r in cpu_rows_all]
        lp = log10.(pv_c); lt = log10.(tv_c)
        n_fit = length(lp)
        b_fit = (n_fit*sum(lp.*lt) - sum(lp)*sum(lt)) / (n_fit*sum(lp.^2) - sum(lp)^2)
        a_fit = (sum(lt) - b_fit*sum(lp)) / n_fit
        t_extrap = 10^(a_fit + b_fit*log10(Float64(rw.p)))
        @sprintf("~%.1fx", t_extrap/rw.t_gpu)
    else
        "GPU-only"
    end
    @printf("%-6d  %-10d  %-10.1f  %-12s  %-14s  %.4f  %.2e\n",
            rw.p, rw.D, rw.C_MB, t_cpu_str, spd_str, rw.t_gpu, rw.err_gpu)
end
println("-"^90)

for rw in rows
    if rw.t_gpu <= TIME_LIMIT
        @printf("Largest p within %.0f s: p=%d  (t=%.2f s,  D=%d,  C=%.0f MB)\n",
                TIME_LIMIT, rw.p, rw.t_gpu, rw.D, rw.C_MB)
    end
end
println("\nCSV → $csv_path")
println("PNG → $png_path")
