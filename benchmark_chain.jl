using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra, StaticArrays, SparseArrays, CUDA
using Plots, Printf, DelimitedFiles, Dates

BLAS.set_num_threads(1)

const TIME_LIMIT = 10.0
const ACC_TIME   = 1.0
const MAX_REPS   = 10
const REF_MAX_T  = 5.0

struct BRow
    problem::String; method::String; order::Int; p::Int
    D::Int; t_s::Float64; mem_MB::Float64; rho::Float64
end

function D_of_mf(rst, d)
    r = div(rst.n, d) - 1
    StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
end

function append_csv(path, row::BRow)
    new_file = !isfile(path)
    open(path, "a") do io
        new_file && println(io, "problem,method,order,p,D,t_s,mem_MB,rho")
        @printf(io, "%s,%s,%d,%d,%d,%.6f,%.4f,%.10f\n",
                row.problem, row.method, row.order, row.p,
                row.D, row.t_s, row.mem_MB, row.rho)
    end
end

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

    p1 = plot(title="Time vs p",      xscale=:log10, yscale=:log10,
              xlabel="p", ylabel="Wall time (s)", legend=:topleft,  margin=5Plots.mm)
    p2 = plot(title="Error vs p",     xscale=:log10, yscale=:log10,
              xlabel="p", ylabel="|ρ−ρ_ref|/ρ_ref", legend=:topright, margin=5Plots.mm)
    p3 = plot(title="Memory vs p",    xscale=:log10, yscale=:log10,
              xlabel="p", ylabel="Memory (MB)", legend=:topleft,  margin=5Plots.mm)
    p4 = plot(title="Work-precision", xscale=:log10, yscale=:log10,
              xlabel="Wall time (s)", ylabel="|ρ−ρ_ref|/ρ_ref", legend=:topright, margin=5Plots.mm)

    for order in [0,1,2]
        ρr = ref_rho_for(all_rows, pname, order)
        for mkey in ["orig","mf","gpu3"]
            sty = METHOD_STYLE[mkey]
            sub = sort(filter(r -> r.method==mkey && r.order==order && r.problem==pname, rows), by=r->r.p)
            isempty(sub) && continue
            pv = Float64.([r.p for r in sub])
            tv = Float64.([r.t_s for r in sub])
            ev = [isnan(ρr)||abs(ρr)<1e-15 ? 1e-14 : max(abs(r.rho-ρr)/abs(ρr),1e-14) for r in sub]
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
    println("  → plot: $png_path")
end

# ── Benchmark loop ─────────────────────────────────────────────────────────
function run_benchmark!(all_rows, pname, lddep, disc_length, ps_grid,
                        csv_path, png_path, title_str; orders=[0,1,2])
    d = size(lddep.A, 2)
    println("\n", "═"^72)
    println("Problem: $pname  (d=$d)")
    println("═"^72)

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
        stopped = Dict("orig"=>false,"mf"=>false,"gpu3"=>false)

        for p in ps_grid
            all(values(stopped)) && break
            method = SemiDiscretization(order, disc_length/p)
            rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, disc_length)

            for (mkey, fn_probe, fn_rep, get_D) in [
                ("orig",
                 () -> spectralRadiusOfMapping(DiscreteMapping_M2(rst)),
                 () -> spectralRadiusOfMapping(DiscreteMapping_M2(rst)),
                 () -> rst.n^2),
                ("mf",
                 () -> spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst)),
                 () -> spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst)),
                 () -> D_of_mf(rst, d)),
                ("gpu3",
                 () -> (r=spectralRadiusOfMapping_GPU_v3(DiscreteMapping_M2_MF(rst)); CUDA.synchronize(); r),
                 () -> (r=spectralRadiusOfMapping_GPU_v3(DiscreteMapping_M2_MF(rst)); CUDA.synchronize(); r),
                 () -> D_of_mf(rst, d)),
            ]
                stopped[mkey] && continue
                t1 = @elapsed rho = fn_probe()
                if t1 > TIME_LIMIT
                    stopped[mkey] = true
                    @printf("    [%-4s]  p=%5d  STOP (%.1fs)\n", mkey, p, t1)
                    continue
                end
                n   = max(1, min(MAX_REPS, ceil(Int, ACC_TIME/t1)))
                t_s = n > 1 ? (@elapsed for _ in 1:n; fn_rep(); end)/n : t1
                mem = @allocated fn_probe()
                D   = get_D()
                row = BRow(pname, mkey, order, p, D, t_s, mem/1024^2, rho)
                push!(all_rows, row); append_csv(csv_path, row)
                @printf("    [%-4s]  p=%5d D=%8d t=%8.4fs mem=%7.1fMB ρ=%.8f\n",
                        mkey, p, D, t_s, mem/1024^2, rho)
                t_s > TIME_LIMIT && (stopped[mkey] = true)
            end
            update_plot(all_rows, pname, title_str, png_path)
        end

        # Reference
        best_rows = filter(r -> r.problem==pname && r.order==order &&
                                r.method in ("mf","gpu3"), all_rows)
        if !isempty(best_rows)
            p_best = maximum(r.p for r in best_rows)
            p_ref  = round(Int, p_best * 1.5)
            println("\n  Reference at p=$p_ref (MF, order=$order)...")
            try
                ref_m   = SemiDiscretization(order, disc_length/p_ref)
                ref_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, ref_m, disc_length)
                ref_dmf = DiscreteMapping_M2_MF(ref_rst)
                t_r = @elapsed ρr = spectralRadiusOfMapping_MF(ref_dmf)
                if t_r <= REF_MAX_T
                    D_r = D_of_mf(ref_rst, d)
                    row = BRow(pname,"ref",order,p_ref,D_r,t_r,0.0,ρr)
                    push!(all_rows, row); append_csv(csv_path, row)
                    @printf("  ref: p=%d  ρ=%.10f  (%.2fs)\n", p_ref, ρr, t_r)
                    update_plot(all_rows, pname, title_str, png_path)
                else
                    @printf("  ref p=%d too slow (%.1fs); using p=%d\n", p_ref, t_r, p_best)
                end
            catch e; println("  ref failed: $e"); end
        end
    end
end

# ── 10-DOF chain setup ─────────────────────────────────────────────────────
println("GPU: ", CUDA.name(CUDA.device()))

n_dof=10; d_chain=20
m_v=1.0; k_v=10.0; c_v=0.3; kp=5.0; kd=1.0; σ_ch=0.02

K_arr = diagm(0=>fill(2k_v,n_dof), 1=>fill(-k_v,n_dof-1), -1=>fill(-k_v,n_dof-1))
K_arr[1,1]=k_v; K_arr[n_dof,n_dof]=k_v
C_arr = (c_v/k_v).*K_arr
M_inv_v = 1.0/m_v

A_arr = [zeros(n_dof,n_dof) I; -M_inv_v.*K_arr -M_inv_v.*C_arr]
A_sys = SMatrix{d_chain,d_chain}(A_arr)

B_arr = zeros(d_chain,d_chain)
B_arr[2*n_dof, n_dof]     = -M_inv_v*kp
B_arr[2*n_dof, 2*n_dof]   = -M_inv_v*kd
B_del = SMatrix{d_chain,d_chain}(B_arr)

sv_ch   = SVector{d_chain}([zeros(2*n_dof-1); σ_ch])
σVec_ch = stAdditive(1, Additive(sv_ch))
z_ch    = SMatrix{d_chain,d_chain}(zeros(d_chain,d_chain))
αMx_ch  = stCoeffMX(1, ProportionalMX(z_ch))
AMx_ch  = ProportionalMX(A_sys)
cVec_ch = Additive(d_chain)

# Find the existing CSV to append to
csv_files = filter(f -> startswith(f,"benchmark_suite_") && endswith(f,".csv"),
                   readdir("."))
ts_fallback = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
csv_path = isempty(csv_files) ? "benchmark_suite_$(ts_fallback).csv" :
           joinpath(".", sort(csv_files)[end])
println("Appending to CSV: $csv_path")

timestamp = replace(splitext(basename(csv_path))[1], "benchmark_suite_" => "")
all_rows = BRow[]   # fresh for plotting (CSV already has 1a/1b/1c)

ps_50 = unique(round.(Int, exp10.(range(log10(10), log10(5000), length=50))))

for (tag, τ_val) in [("2a", 0.3), ("2b", 0.7), ("2c", 1.5)]
    pname = "chain10dof_$(tag)"
    png   = "benchmark_$(tag)_$(timestamp).png"
    title = "10-DOF chain PD control ($(tag): τ=$(τ_val)s)  [$(CUDA.name(CUDA.device()))]"

    βMx_i = stCoeffMX(1, DelayMX(τ_val, z_ch))
    BMx_i = DelayMX(τ_val, B_del)
    lddep = LDDEProblem(AMx_ch, [BMx_i], [αMx_ch], [βMx_i], cVec_ch, [σVec_ch])

    run_benchmark!(all_rows, pname, lddep, τ_val, ps_50, csv_path, png, title)
end

println("\nDone. CSV: $csv_path")
