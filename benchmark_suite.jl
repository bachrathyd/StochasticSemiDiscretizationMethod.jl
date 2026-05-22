using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra, StaticArrays, SparseArrays, CUDA
using Plots, Printf, DelimitedFiles, Dates

BLAS.set_num_threads(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
const TIME_LIMIT   = 10.0   # stop if single evaluation exceeds this (seconds)
const ACC_TIME     = 1.0    # accumulate this much measurement time for fast runs
const MAX_REPS     = 10     # but never more repeats than this
const REF_MAX_T    = 5.0    # max seconds allowed for a reference computation

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCT
# ══════════════════════════════════════════════════════════════════════════════
struct BRow
    problem::String
    method::String   # "orig" | "mf" | "gpu3" | "ref"
    order::Int
    p::Int
    D::Int
    t_s::Float64
    mem_MB::Float64
    rho::Float64
end

function D_of_mf(rst, d)
    r = div(rst.n, d) - 1
    StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
end

# ══════════════════════════════════════════════════════════════════════════════
# CSV  (append, create header on first write)
# ══════════════════════════════════════════════════════════════════════════════
function append_csv(path, row::BRow)
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, "problem,method,order,p,D,t_s,mem_MB,rho")
        @printf(io, "%s,%s,%d,%d,%d,%.6f,%.4f,%.10f\n",
                row.problem, row.method, row.order, row.p,
                row.D, row.t_s, row.mem_MB, row.rho)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING  (reads all_rows; writes png every time it's called)
# ══════════════════════════════════════════════════════════════════════════════
const METHOD_STYLE = Dict(
    "orig" => (label="Orig (matrix)", color=:red,   ls=:solid, mk=:square),
    "mf"   => (label="MF-CPU",        color=:blue,  ls=:solid, mk=:circle),
    "gpu3" => (label="GPU v3",        color=:green, ls=:solid, mk=:diamond),
)
const ORDER_DASH = Dict(0=>:solid, 1=>:dash, 2=>:dot)

function ref_rho_for(all_rows, pname, order)
    ref_rows = filter(r -> r.problem==pname && r.order==order && r.method=="ref", all_rows)
    !isempty(ref_rows) && return ref_rows[end].rho
    cands = filter(r -> r.problem==pname && r.order==order && r.method in ("mf","gpu3"), all_rows)
    isempty(cands) && return NaN
    cands[argmax([r.p for r in cands])].rho
end

function update_plot(all_rows, pname, title_str, png_path)
    rows = filter(r -> r.problem==pname && r.method != "ref", all_rows)
    isempty(rows) && return

    p1 = plot(title="Time vs p",        xscale=:log10, yscale=:log10,
              xlabel="p", ylabel="Wall time (s)", legend=:topleft,  margin=5Plots.mm)
    p2 = plot(title="Error vs p",       xscale=:log10, yscale=:log10,
              xlabel="p", ylabel="|ρ−ρ_ref|/ρ_ref", legend=:topright, margin=5Plots.mm)
    p3 = plot(title="Memory vs p",      xscale=:log10, yscale=:log10,
              xlabel="p", ylabel="Memory (MB)", legend=:topleft,  margin=5Plots.mm)
    p4 = plot(title="Work-precision",   xscale=:log10, yscale=:log10,
              xlabel="Wall time (s)", ylabel="|ρ−ρ_ref|/ρ_ref", legend=:topright, margin=5Plots.mm)

    for order in [0,1,2]
        ρr = ref_rho_for(all_rows, pname, order)
        for mkey in ["orig","mf","gpu3"]
            sty = METHOD_STYLE[mkey]
            sub = sort(filter(r -> r.method==mkey && r.order==order && r.problem==pname, rows),
                       by=r->r.p)
            isempty(sub) && continue
            pv = Float64.([r.p    for r in sub])
            tv = Float64.([r.t_s  for r in sub])
            ev = [isnan(ρr)||abs(ρr)<1e-15 ? 1e-14 :
                  max(abs(r.rho-ρr)/abs(ρr), 1e-14) for r in sub]
            mv = Float64.([r.mem_MB for r in sub])
            lbl = "$(sty.label) ord$order"
            ls  = ORDER_DASH[order]
            plot!(p1, pv, tv, label=lbl, color=sty.color, ls=ls, marker=sty.mk, ms=4, lw=1.5)
            plot!(p2, pv, ev, label=lbl, color=sty.color, ls=ls, marker=sty.mk, ms=4, lw=1.5)
            plot!(p3, pv, mv, label=lbl, color=sty.color, ls=ls, marker=sty.mk, ms=4, lw=1.5)
            plot!(p4, tv, ev, label=lbl, color=sty.color, ls=ls, marker=sty.mk, ms=4, lw=1.5)
        end
    end
    hline!(p1, [TIME_LIMIT], ls=:dash, color=:black, lw=1.5, label="10s limit")

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1500,1100),
               plot_title=title_str, margin=6Plots.mm)
    savefig(fig, png_path)
    println("  → plot updated: $png_path")
end

# ══════════════════════════════════════════════════════════════════════════════
# CORE BENCHMARK FOR ONE PROBLEM
# ══════════════════════════════════════════════════════════════════════════════
function run_benchmark!(all_rows, pname, lddep, disc_length, ps_grid,
                        csv_path, png_path, title_str; orders=[0,1,2])
    d = size(lddep.A, 2)

    println("\n", "═"^72)
    println("Problem: $pname  (d=$d, disc_len=$(round(disc_length,digits=4)))")
    println("═"^72)

    # JIT warm-up
    println("  JIT warm-up (p=20)...")
    _m  = SemiDiscretization(1, disc_length/20)
    _r  = StochasticSemiDiscretizationMethod.calculateResults(lddep, _m, disc_length)
    _mf = DiscreteMapping_M2_MF(_r)
    _m2 = DiscreteMapping_M2(_r)
    spectralRadiusOfMapping(_m2)
    spectralRadiusOfMapping_MF(_mf)
    spectralRadiusOfMapping_GPU_v3(_mf); CUDA.synchronize()
    println("  done.")

    for order in orders
        println("\n  ── Order $order ──")
        stopped = Dict("orig"=>false, "mf"=>false, "gpu3"=>false)

        for p in ps_grid
            all(values(stopped)) && break
            method = SemiDiscretization(order, disc_length/p)
            rst    = StochasticSemiDiscretizationMethod.calculateResults(
                         lddep, method, disc_length)

            # ── orig ──────────────────────────────────────────────────────
            if !stopped["orig"]
                dm2  = DiscreteMapping_M2(rst)
                D_o  = rst.n^2
                t1   = @elapsed rho = spectralRadiusOfMapping(dm2)
                if t1 > TIME_LIMIT
                    stopped["orig"] = true
                    @printf("    [orig]  p=%5d  STOP (%.1fs)\n", p, t1)
                else
                    n = max(1, min(MAX_REPS, ceil(Int, ACC_TIME/t1)))
                    t_s = if n > 1
                        (@elapsed for _ in 1:n; spectralRadiusOfMapping(dm2); end) / n
                    else; t1; end
                    mem = @allocated spectralRadiusOfMapping(dm2)
                    row = BRow(pname,"orig",order,p,D_o,t_s,mem/1024^2,rho)
                    push!(all_rows, row); append_csv(csv_path, row)
                    @printf("    [orig]  p=%5d D=%8d t=%8.4fs mem=%7.1fMB ρ=%.8f\n",
                            p,D_o,t_s,mem/1024^2,rho)
                    t_s > TIME_LIMIT && (stopped["orig"]=true)
                end
            end

            # ── mf ────────────────────────────────────────────────────────
            if !stopped["mf"]
                dmf  = DiscreteMapping_M2_MF(rst)
                D_m  = D_of_mf(rst, d)
                t1   = @elapsed rho = spectralRadiusOfMapping_MF(dmf)
                if t1 > TIME_LIMIT
                    stopped["mf"] = true
                    @printf("    [mf]    p=%5d  STOP (%.1fs)\n", p, t1)
                else
                    n = max(1, min(MAX_REPS, ceil(Int, ACC_TIME/t1)))
                    t_s = if n > 1
                        (@elapsed for _ in 1:n; spectralRadiusOfMapping_MF(dmf); end) / n
                    else; t1; end
                    mem = @allocated spectralRadiusOfMapping_MF(dmf)
                    row = BRow(pname,"mf",order,p,D_m,t_s,mem/1024^2,rho)
                    push!(all_rows, row); append_csv(csv_path, row)
                    @printf("    [mf]    p=%5d D=%8d t=%8.4fs mem=%7.1fMB ρ=%.8f\n",
                            p,D_m,t_s,mem/1024^2,rho)
                    t_s > TIME_LIMIT && (stopped["mf"]=true)
                end
            end

            # ── gpu3 ──────────────────────────────────────────────────────
            if !stopped["gpu3"]
                dmf  = DiscreteMapping_M2_MF(rst)
                D_g  = D_of_mf(rst, d)
                t1   = @elapsed begin
                    rho = spectralRadiusOfMapping_GPU_v3(dmf); CUDA.synchronize()
                end
                if t1 > TIME_LIMIT
                    stopped["gpu3"] = true
                    @printf("    [gpu3]  p=%5d  STOP (%.1fs)\n", p, t1)
                else
                    n = max(1, min(MAX_REPS, ceil(Int, ACC_TIME/t1)))
                    t_s = if n > 1
                        (@elapsed for _ in 1:n
                            spectralRadiusOfMapping_GPU_v3(dmf); CUDA.synchronize()
                        end) / n
                    else; t1; end
                    mem = @allocated spectralRadiusOfMapping_GPU_v3(dmf)
                    row = BRow(pname,"gpu3",order,p,D_g,t_s,mem/1024^2,rho)
                    push!(all_rows, row); append_csv(csv_path, row)
                    @printf("    [gpu3]  p=%5d D=%8d t=%8.4fs mem=%7.1fMB ρ=%.8f\n",
                            p,D_g,t_s,mem/1024^2,rho)
                    t_s > TIME_LIMIT && (stopped["gpu3"]=true)
                end
            end

            # live plot after each p
            update_plot(all_rows, pname, title_str, png_path)
        end  # ps_grid

        # ── reference: one step beyond the furthest point ─────────────────
        best_rows = filter(r -> r.problem==pname && r.order==order &&
                                r.method in ("mf","gpu3"), all_rows)
        if !isempty(best_rows)
            p_best = maximum(r.p for r in best_rows)
            p_ref  = round(Int, p_best * 1.5)
            println("\n  Reference at p=$p_ref (MF, order=$order)...")
            try
                ref_m   = SemiDiscretization(order, disc_length/p_ref)
                ref_rst = StochasticSemiDiscretizationMethod.calculateResults(
                              lddep, ref_m, disc_length)
                ref_dmf = DiscreteMapping_M2_MF(ref_rst)
                t_r, ρr = @elapsed(spectralRadiusOfMapping_MF(ref_dmf)), 0.0
                t_r = @elapsed ρr = spectralRadiusOfMapping_MF(ref_dmf)
                if t_r <= REF_MAX_T
                    D_r = D_of_mf(ref_rst, d)
                    row = BRow(pname,"ref",order,p_ref,D_r,t_r,0.0,ρr)
                    push!(all_rows, row); append_csv(csv_path, row)
                    @printf("  ref: p=%d  ρ=%.10f  (%.2fs)\n", p_ref, ρr, t_r)
                    update_plot(all_rows, pname, title_str, png_path)
                else
                    @printf("  ref p=%d too slow (%.1fs); using p=%d as ref\n",
                            p_ref, t_r, p_best)
                end
            catch e
                println("  ref failed: $e")
            end
        end
    end  # orders
end

# ══════════════════════════════════════════════════════════════════════════════
# p grid
# ══════════════════════════════════════════════════════════════════════════════
ps_50 = unique(round.(Int, exp10.(range(log10(10), log10(5000), length=50))))

timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
csv_path  = "benchmark_suite_$(timestamp).csv"
all_rows  = BRow[]

println("GPU: ", CUDA.name(CUDA.device()))
println("CSV: $csv_path\n")

# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM SET 1: Stochastic Mathieu with TWO DELAYS (τ2 time-varying)
# τ1 = 2π (constant)
# τ2(t) = τ1 * (1 + 0.5*sin(2π/T * t))   max = τ1*1.5
# T ∈ {0.2π, 2π, 20π}
# ══════════════════════════════════════════════════════════════════════════════
println("\n" * "█"^72)
println("PROBLEM SET 1: Stochastic Mathieu, 2 delays (τ2 time-varying)")
println("█"^72)

τ1_m = 2π
A_m=1.0; ε_m=0.5; B1_m=0.15; B2_m=0.1; ζ_m=0.1; σ_m=0.05; α_m=0.1
P_math = 2π
# disc_length = max delay = τ1 * 1.5
τ_max_math = τ1_m * 1.5

B1m_s = @SMatrix [0.0 0.0; B1_m 0.0]
B2m_s = @SMatrix [0.0 0.0; B2_m 0.0]
am_s  = @SMatrix [0.0 0.0; α_m  0.0]
zm_s  = @SMatrix [0.0 0.0; 0.0  0.0]
sv_m  = @SVector [0.0, σ_m]

for (tag, T_val) in [("1a", 0.2π), ("1b", 2π), ("1c", 20π)]
    T_label = "$(round(T_val/π, digits=2))π"
    pname   = "mathieu2d_$(tag)"
    png     = "benchmark_$(tag)_$(timestamp).png"
    title   = "Stochastic Mathieu 2-delay ($(tag): T=$(T_label))  [$(CUDA.name(CUDA.device()))]"

    τ2fun(t) = τ1_m * (1.0 + 0.5*sin(2π/T_val * t))
    AMxfun(t) = @SMatrix [0.0 1.0; -(A_m + ε_m*cos(2π*t/P_math)) -2ζ_m]
    AMx   = ProportionalMX(AMxfun)
    BMx1  = DelayMX(τ1_m, B1m_s)
    BMx2  = DelayMX(τ2fun, B2m_s)
    αMx1  = stCoeffMX(1, ProportionalMX(am_s))
    βMx1  = stCoeffMX(1, DelayMX(τ1_m, zm_s))
    σVec  = stAdditive(1, Additive(sv_m))
    cVec  = Additive(2)
    lddep = LDDEProblem(AMx, [BMx1, BMx2], [αMx1], [βMx1], cVec, [σVec])

    run_benchmark!(all_rows, pname, lddep, τ_max_math, ps_50,
                   csv_path, png, title)
end

# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM SET 2: 10-DOF mass-spring-damper chain + delayed PD control
# Longitudinal wave, left end pinned, PD force on right end with delay τ
# State: [q1…q10, dq̇1…dq̇10]  (d=20)
# ══════════════════════════════════════════════════════════════════════════════
println("\n" * "█"^72)
println("PROBLEM SET 2: 10-DOF mass-spring-damper + PD delay control")
println("█"^72)

n_dof   = 10
d_chain = 2*n_dof   # = 20
m_v=1.0; k_v=10.0; c_v=0.3; kp=5.0; kd=1.0; σ_ch=0.02

# Stiffness matrix: chain, left pinned, right free
K_arr = diagm(0 => fill(2k_v, n_dof),
               1 => fill(-k_v, n_dof-1),
              -1 => fill(-k_v, n_dof-1))
K_arr[1, 1]       = k_v   # mass 1: only spring to wall (left) + spring to mass 2
K_arr[n_dof,n_dof]= k_v   # mass 10: only spring to mass 9 (free right)
C_arr = (c_v/k_v) .* K_arr
M_inv_v = 1.0/m_v          # scalar (uniform masses)

A_top = zeros(n_dof, n_dof)
A_arr = [A_top I; -M_inv_v.*K_arr  -M_inv_v.*C_arr]
A_sys = SMatrix{d_chain,d_chain}(A_arr)

# Delay matrix: PD control on mass n_dof
B_arr = zeros(d_chain, d_chain)
B_arr[n_dof+n_dof, n_dof]       = -M_inv_v * kp   # position feedback
B_arr[n_dof+n_dof, n_dof+n_dof] = -M_inv_v * kd   # velocity feedback
B_del = SMatrix{d_chain,d_chain}(B_arr)

# Noise: small additive on tip velocity only
sv_ch   = SVector{d_chain}([zeros(2*n_dof-1); σ_ch])
σVec_ch = stAdditive(1, Additive(sv_ch))
z_ch    = SMatrix{d_chain,d_chain}(zeros(d_chain,d_chain))
αMx_ch  = stCoeffMX(1, ProportionalMX(z_ch))
βMx_ch  = stCoeffMX(1, DelayMX(1.0, z_ch))   # placeholder τ — overridden per case
cVec_ch = Additive(d_chain)
AMx_ch  = ProportionalMX(A_sys)

for (tag, τ_val) in [("2a", 0.3), ("2b", 0.7), ("2c", 1.5)]
    pname   = "chain10dof_$(tag)"
    png     = "benchmark_$(tag)_$(timestamp).png"
    title   = "10-DOF chain PD control ($(tag): τ=$(τ_val)s)  [$(CUDA.name(CUDA.device()))]"

    # Rebuild βMx with correct τ
    βMx_i = stCoeffMX(1, DelayMX(τ_val, z_ch))
    BMx_i = DelayMX(τ_val, B_del)
    lddep = LDDEProblem(AMx_ch, [BMx_i], [αMx_ch], [βMx_i], cVec_ch, [σVec_ch])

    run_benchmark!(all_rows, pname, lddep, τ_val, ps_50,
                   csv_path, png, title)
end

println("\n\nAll benchmarks complete.")
println("CSV: $csv_path")
