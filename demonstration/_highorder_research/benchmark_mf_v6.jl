# =============================================================================
# benchmark_mf_v6.jl  —  MIRROR of benchmark_mf_complexity.jl (same problem, setup, layout, slope)
# but EXTENDED with the new v6 high-order moment-collocation GL(1)..GL(6) curves alongside the
# trusted SDM orders 0,1,2. Identical Mathieu params, BLAS=1 thread, ρ_ref = SDM q=2 p=800.
# Goal: see the v6 curves on the SAME, smooth, log-log Mathieu where SDM forms straight lines —
# so we can compare ORDER cleanly (slope = order). Mirrors slope_log10 / second-half average.
# =============================================================================
using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using BenchmarkTools, StaticArrays, LinearAlgebra, Plots, Printf, Dates

BLAS.set_num_threads(1)   # same as the reference benchmark

# ------------ Identical Mathieu problem ------------------------------------------------
function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val, P)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε*cos(2π*t/P)) -2ζ]
    AMx  = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2)
    αMx1  = stCoeffMX(1, ProportionalMX(@SMatrix [0. 0.; α_val 0.]))
    βMx11 = stCoeffMX(1, DelayMX(τ, @SMatrix [0. 0.; α_val 0.]))   # β=α_val (matches updated reference)
    σVec  = stAdditive(1, Additive(@SVector [0., σ]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end
const A=1.0; const ε=0.5; const B=0.2; const ζ=0.1; const τ=1.0; const σ=0.1; const α_val=0.2; const P=1.0
lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val, P)

# Reference (high-res SDM order-2), same as the original benchmark
println("Computing reference ρ (SDM q=2, p=800)...")
ref_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, SemiDiscretization(2, P/800), τ)
const ρ_ref = spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(ref_rst))
@printf("  ρ_ref = %.10f\n\n", ρ_ref)

# ------------ v6 problem on the SAME Mathieu (NB: this benchmark has β=0, only present-α mult. noise) ----
# Map the SSDM-style problem into the v6 Prob struct (cov_colloc_v6.jl).
include(joinpath(@__DIR__,"demonstration","cov_colloc_v6.jl"))
FILL_OFFDIAG[]=false; CROSS_ON[]=true
Amat_v6(t)=[0.0 1.0; -(A + ε*cos(2π*t/P)) -2ζ]
Bmat_v6(t)=[0.0 0.0; B 0.0]
αmat_v6(t)=[0.0 0.0; α_val 0.0]
βmat_v6(t)=[0.0 0.0; α_val 0.0]                       # β=α_val (delayed multipl. noise present — matches updated ref)
pb_v6 = Prob(2, P, τ, Amat_v6, Bmat_v6, αmat_v6, βmat_v6)
# v6 doesn't model additive σ (it tracks the homogeneous 2nd-moment map ρ); same as M2_MF, the σ
# does not affect ρ — both compute ρ of the homogeneous covariance recurrence.

# ------------ Sweep grid (same 25-point log range from p=4..500 as the ref benchmark) ----
ps_all = unique(round.(Int, exp10.(range(log10(4), log10(500), length=25))))
const TIME_LIMIT = 3.0

# slope helper — IDENTICAL to the reference benchmark (log10, second-half average of consecutive slopes)
function log_slope(xs, ys)
    valid = (xs .> 0) .& (ys .> 1e-15)
    lx = log10.(xs[valid]); ly = log10.(ys[valid])
    length(lx) < 3 && return NaN
    n = length(lx); i0 = max(1, n÷2)
    lx = lx[i0:end]; ly = ly[i0:end]
    slopes = diff(ly) ./ diff(lx)
    return round(sum(slopes)/length(slopes), digits=2)
end

# ------------ Bench rows ----
struct BenchRow
    method::String; nominal_order::Int; p::Int; D::Int
    time_s::Float64; mem_MB::Float64; rho::Float64; err::Float64
end
all_rows = BenchRow[]

# ------------ Per-method timing functions ----
function bench_sdm(order, p)
    method = SemiDiscretization(order, P/p)
    rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm = DiscreteMapping_M2_MF(rst)
    d=2; r = div(rst.n, d) - 1
    D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
    ρ = spectralRadiusOfMapping_MF(dm)         # warm-up
    bm = @benchmark spectralRadiusOfMapping_MF($dm) samples=3 evals=1 seconds=1.0
    return D, median(bm).time/1e9, bm.memory/1024^2, ρ
end
function bench_v6(S, p)
    eng = build_v6(pb_v6, S, p)                 # build outside the timed kernel (matches SDM: dm built once)
    D = (size(eng.U,1)*(size(eng.U,1)+1))÷2     # symmetric-covariance vector length (the spectral search space)
    ρ = rho_H_krylov(eng)                       # warm-up
    bm = @benchmark rho_H_krylov($eng) samples=3 evals=1 seconds=1.0
    return D, median(bm).time/1e9, bm.memory/1024^2, ρ
end

# ------------ Output paths ----
ts_stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
csv_path = "benchmark_mf_v6_$(ts_stamp).csv"
png_path = "benchmark_mf_v6_$(ts_stamp).png"
println("Output files:\n  CSV → $csv_path\n  PNG → $png_path\n")

# distinct colors/markers for SDM + GL
methcfg = [
    ("SDM0",0, :blue,     :circle  ),
    ("SDM1",1, :red,      :rect    ),
    ("SDM2",2, :green,    :diamond ),
    ("GL1", 2, :dodgerblue,:utriangle),    # nominal 2S=2
    ("GL2", 4, :seagreen, :utriangle),     # nominal 2S=4
    ("GL3", 6, :purple,   :utriangle),     # nominal 2S=6
    ("GL4", 8, :darkorange,:utriangle),    # nominal 2S=8
    ("GL5",10, :red,      :utriangle),     # nominal 2S=10
    ("GL6",12, :magenta,  :utriangle),     # nominal 2S=12
]

function _update_plot(rows, methcfg, png_path)
    p1 = plot(title="Time vs p  (MF-SSDM + v6, Mathieu d=2)",
              xlabel="p  (steps per period)", ylabel="CPU time (s)",
              xscale=:log10, yscale=:log10, legend=:topleft, grid=:both, minorgrid=true, legendfontsize=6)
    p2 = plot(title="Spectral-radius error vs p",
              xlabel="p", ylabel="|ρ - ρ_ref|",
              xscale=:log10, yscale=:log10, legend=:topright, grid=:both, minorgrid=true, legendfontsize=6)
    p3 = plot(title="Operator dimension D vs p",
              xlabel="p", ylabel="D",
              xscale=:log10, yscale=:log10, legend=:topleft, grid=:both, minorgrid=true, legendfontsize=6)
    p4 = plot(title="Work-precision (Error vs CPU time)",
              xlabel="CPU time (s)", ylabel="|ρ - ρ_ref|",
              xscale=:log10, yscale=:log10, legend=:topright, grid=:both, minorgrid=true, legendfontsize=6)
    for (name, _nom, col, mk) in methcfg
        sel = filter(r->r.method==name, rows); isempty(sel)&&continue
        pv=[r.p for r in sel]; tv=[r.time_s for r in sel]; Dv=[r.D for r in sel]; ev=[r.err for r in sel]
        st=log_slope(pv,tv); se=log_slope(pv,ev); sD=log_slope(pv,Dv)
        plot!(p1,pv,tv,marker=mk,color=col,label="$name (sl≈$st)")
        plot!(p2,pv,ev,marker=mk,color=col,label="$name (sl≈$se)")
        plot!(p3,pv,Dv,marker=mk,color=col,label="$name (sl≈$sD)")
        plot!(p4,tv,ev,marker=mk,color=col,label="$name")
    end
    # O(p²)/O(p³) anchors on the Time plot (same as ref)
    ord0 = filter(r->r.method=="SDM0", rows)
    if length(ord0)>=2
        pv=[r.p for r in ord0]; tv=[r.time_s for r in ord0]
        i0=max(1,length(pv)÷2); p0=Float64(pv[i0]); t0=tv[i0]
        pr=Float64.([pv[1],pv[end]])
        plot!(p1,pr, t0 .*(pr./p0).^2, ls=:dash, color=:gray,  lw=2, label="O(p²)")
        plot!(p1,pr, t0 .*(pr./p0).^3, ls=:dot,  color=:black, lw=2, label="O(p³)")
    end
    fig = plot(p1,p2,p3,p4, layout=(2,2), size=(1500,1100),
               plot_title="MF-SSDM (q=0,1,2) + v6 GL(1)..GL(6)  —  Stoch. Mathieu (d=2, τ=$τ)")
    savefig(fig, png_path)
end

# write CSV after every point
function _write_csv(rows, csv_path)
    open(csv_path,"w") do io
        println(io,"method,nominal_order,p,D,time_s,mem_MB,rho,err")
        for r in rows
            @printf(io,"%s,%d,%d,%d,%.6f,%.4f,%.10f,%.6e\n",
                    r.method,r.nominal_order,r.p,r.D,r.time_s,r.mem_MB,r.rho,r.err)
        end
    end
end

# ============= SWEEP =============
function runall(methcfg, ps_all)
    rows = BenchRow[]
    for (name, nom, _col, _mk) in methcfg
        println("=== $name (nominal order $nom) ===")
        for p in ps_all
            try
                D,t,m,ρ = startswith(name,"SDM") ? bench_sdm(parse(Int,name[4:end]), p) : bench_v6(parse(Int,name[3:end]), p)
                err = abs(ρ - ρ_ref)
                @printf("  p=%4d D=%7d t=%8.4fs mem=%7.1fMB ρ=%.8f err=%.2e\n", p,D,t,m,ρ,err)
                push!(rows, BenchRow(name,nom,p,D,t,m,ρ,err))
                _write_csv(rows, csv_path); _update_plot(rows, methcfg, png_path)
                if t > TIME_LIMIT; println("  → time limit, stopping $name"); break; end
            catch e; @warn "$name p=$p" e; end
        end
        println()
    end
    rows
end
all_rows = runall(methcfg, ps_all)
println("Done. CSV → $csv_path\n      PNG → $png_path")
