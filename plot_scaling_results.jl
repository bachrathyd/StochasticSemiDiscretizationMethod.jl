using Pkg; Pkg.activate(".")
using Printf, Plots, DelimitedFiles

# ---- Load data from the cleanest benchmark run ----
csv_path = "benchmark_gpu_scaling_20260520_184921.csv"

data, hdr = readdlm(csv_path, ',', header=true)
hdr = vec(hdr)

col(name) = findfirst(==(name), hdr)
ps   = Int.(data[:, col("p")])
Ds   = Int.(data[:, col("D")])
CMBs = data[:, col("C_MB")]
ρs   = data[:, col("rho_gpu")]
errs = data[:, col("err_gpu")]
tGPU = data[:, col("t_gpu_s")]
tCPU = data[:, col("t_cpu_s")]   # -1 means not measured

ρ_ref = 0.6817666221

# ---- Separate measured and GPU-only rows ----
cpu_mask = tCPU .> 0
gpu_mask = tGPU .> 0

pv     = ps[gpu_mask]
tv_gpu = tGPU[gpu_mask]
Dv     = Ds[gpu_mask]
ev     = errs[gpu_mask]
CMv    = CMBs[gpu_mask]

pv_cpu    = ps[cpu_mask]
tv_cpu    = tCPU[cpu_mask]

TIME_LIMIT = 10.0

# ---- CPU power-law fit (log-log least squares) ----
lp = log10.(Float64.(pv_cpu))
lt = log10.(tv_cpu)
n  = length(lp)
b_fit = (n*sum(lp.*lt) - sum(lp)*sum(lt)) / (n*sum(lp.^2) - sum(lp)^2)
a_fit = (sum(lt) - b_fit*sum(lp)) / n

t_cpu_extrap(p) = 10^(a_fit + b_fit*log10(Float64(p)))

@printf("CPU power-law fit: t_cpu ≈ 10^(%.4f) × p^(%.4f)\n", a_fit, b_fit)
@printf("  (= %.3e × p^%.3f)\n\n", 10^a_fit, b_fit)

# ---- Summary table ----
println("="^95)
@printf("%-6s  %-10s  %-10s  %-16s  %-14s  %-12s  %s\n",
        "p", "D", "C (MB)", "t_CPU (s)", "t_GPU v3 (s)", "speedup", "err")
println("-"^95)
for i in 1:length(pv)
    p_i = pv[i]
    t_g = tv_gpu[i]
    D_i = Dv[i]
    C_i = CMv[i]
    e_i = ev[i]

    cpu_idx = findfirst(==(p_i), pv_cpu)
    if cpu_idx !== nothing
        t_c     = tv_cpu[cpu_idx]
        cpu_str = @sprintf("%-16.4f", t_c)
        spd_str = @sprintf("%.2fx", t_c/t_g)
    else
        t_c     = t_cpu_extrap(p_i)
        cpu_str = @sprintf("~%-15.1f", t_c)
        spd_str = @sprintf("~%.1fx", t_c/t_g)
    end

    flag = t_g > TIME_LIMIT ? " ← OVER LIMIT" : ""
    @printf("%-6d  %-10d  %-10.1f  %-16s  %-14.4f  %-12s  %.2e%s\n",
            p_i, D_i, C_i, cpu_str, t_g, spd_str, e_i, flag)
end
println("="^95)

# Find limit
println()
for i in 1:length(pv)
    if tv_gpu[i] <= TIME_LIMIT
        p_extrap_cpu = t_cpu_extrap(pv[i])
        @printf("p = %4d within 10 s:  t_GPU = %.3f s,  t_CPU_extrap ≈ %.1f s,  speedup ≈ %.1fx\n",
                pv[i], tv_gpu[i], p_extrap_cpu, p_extrap_cpu/tv_gpu[i])
    end
end
println()
@printf("GPU time at p=2500: %.3f s  (largest p tested under 10 s)\n", tv_gpu[end])

# ---- Estimate crossover and limit from power laws ----
gpu_lp = log10.(Float64.(pv[4:end]))  # use middle-to-large p for GPU slope
gpu_lt = log10.(tv_gpu[4:end])
n_g = length(gpu_lp)
bg_fit = (n_g*sum(gpu_lp.*gpu_lt) - sum(gpu_lp)*sum(gpu_lt)) / (n_g*sum(gpu_lp.^2) - sum(gpu_lp)^2)
ag_fit = (sum(gpu_lt) - bg_fit*sum(gpu_lp)) / n_g
@printf("\nGPU power-law fit (p≥%d): t_GPU ≈ 10^(%.4f) × p^(%.4f)\n", pv[4], ag_fit, bg_fit)

# Solve 10^ag × p^bg = TIME_LIMIT for p
p_limit = 10^((log10(TIME_LIMIT) - ag_fit) / bg_fit)
@printf("Extrapolated 10 s time-limit:  p_max ≈ %.0f\n", p_limit)

# ---- Plot ----
ts = replace(splitext(basename(csv_path))[1], "benchmark_gpu_scaling_" => "")
png_path = "scaling_results_$(ts).png"

# Extended extrapolation range
p_ext = 10 .^ range(log10(10.0), log10(5000.0), length=200)
t_cpu_line = t_cpu_extrap.(p_ext)
t_gpu_line = 10^ag_fit .* p_ext .^ bg_fit

p1 = plot(pv, tv_gpu, marker=:circle, color=:blue, lw=2, ms=5,
          label="GPU v3 (measured, slope≈$(round(bg_fit,digits=2)))",
          title="Compute time vs p", xlabel="p", ylabel="time (s)",
          xscale=:log10, yscale=:log10, legend=:topleft)
plot!(p1, Float64.(pv_cpu), tv_cpu, marker=:square, color=:red, lw=2, ms=5,
      label="CPU MF (measured, slope≈$(round(b_fit,digits=2)))")
plot!(p1, p_ext, t_cpu_line, ls=:dash, color=:red, lw=1.5,
      label="CPU MF (extrapolated)")
plot!(p1, p_ext, t_gpu_line, ls=:dot, color=:blue, lw=1,
      label="GPU v3 (fit)")
hline!(p1, [TIME_LIMIT], ls=:dash, color=:black, lw=1.5, label="10 s limit")
vline!(p1, [p_limit], ls=:dot, color=:gray, lw=1, label="p_max≈$(round(Int,p_limit))")

p2 = plot(Float64.(pv), ev, marker=:circle, color=:blue, lw=2,
          label="GPU v3",
          title="Error vs p  (ref = 0.68177)",
          xlabel="p", ylabel="|ρ − ρ_ref| / ρ_ref",
          xscale=:log10, yscale=:log10, legend=:topright)
# reference: error scales as p^(-2) for second-order SSD
p_ref_line = [Float64(pv[1]), Float64(pv[end])]
e0 = ev[1]
plot!(p2, p_ref_line, e0 .* (p_ref_line./Float64(pv[1])).^(-2), ls=:dash, color=:gray, lw=1, label="O(p⁻²)")

# Speedup plot
cpu_dict     = Dict(zip(pv_cpu, tv_cpu))
spd_measured = [cpu_dict[p] / tv_gpu[i] for (i,p) in enumerate(pv) if p in pv_cpu]
p_spd_meas   = [p for p in pv if p in pv_cpu]
spd_extrap   = [t_cpu_extrap(p) / tv_gpu[i] for (i,p) in enumerate(pv) if !(p in pv_cpu)]
p_spd_ext    = [p for p in pv if !(p in pv_cpu)]

p3 = plot(Float64.(p_spd_meas), spd_measured, marker=:square, color=:green, lw=2, ms=5,
          label="Speedup (measured CPU)",
          title="GPU v3 speedup vs CPU MF",
          xlabel="p", ylabel="speedup (×)", xscale=:log10, yscale=:log10, legend=:topleft)
if !isempty(p_spd_ext)
    plot!(p3, Float64.(p_spd_ext), spd_extrap, marker=:circle, ls=:dash, color=:darkgreen, lw=1.5, ms=4,
          label="Speedup (extrapolated CPU)")
end
hline!(p3, [1.0], ls=:dash, color=:gray, lw=1, label="breakeven")

p4 = plot(Float64.(pv), Dv, marker=:circle, color=:purple, lw=2,
          label="D (state dim)",
          title="State-space dim D vs p", xlabel="p", ylabel="D",
          xscale=:log10, yscale=:log10, legend=:topleft)
plot!(p4, Float64.(pv), CMv, marker=:diamond, color=:orange, lw=2, label="C matrix (MB)")

fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1400,1050),
           plot_title="GPU v3 Scaling — Stochastic Mathieu d=2, order=1  [Quadro P4000]")
savefig(fig, png_path)
println("PNG → $png_path")
