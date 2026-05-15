using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using BenchmarkTools
using StaticArrays
using LinearAlgebra
using Plots
using Printf
using Dates

# Force single-threaded BLAS to avoid sudden overhead jump at p~75
# (Julia switches to multithreaded BLAS for larger matrices which adds
#  thread-launch overhead that dominates for the small d×d matrices here)
BLAS.set_num_threads(1)

# Stochastic delayed Mathieu equation (d=2, standard test case)
function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val, P)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε*cos(2π*t/P)) -2ζ]
    AMx  = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2)
    αMx1  = stCoeffMX(1, ProportionalMX(@SMatrix [0. 0.; α_val 0.]))
    βMx11 = stCoeffMX(1, DelayMX(τ, @SMatrix [0. 0.; 0. 0.]))
    σVec  = stAdditive(1, Additive(@SVector [0., σ]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

A=1.0; ε=0.5; B=0.2; ζ=0.1; τ=1.0; σ=0.1; α_val=0.2; P=1.0
lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val, P)

# Reference (high-res order-2)
println("Computing reference ρ...")
ref_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, SemiDiscretization(2, P/500), τ)
ρ_ref   = spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(ref_rst))
@printf("  ρ_ref = %.10f\n\n", ρ_ref)

orders = [0, 1, 2]
# 25 points log-spaced from p=4 to p=800
ps_all = unique(round.(Int, exp10.(range(log10(4), log10(500), length=25))))

TIME_LIMIT = 3.0   # seconds — stop an order when exceeded

struct BenchRow
    order::Int; p::Int; D::Int
    time_s::Float64; mem_MB::Float64
    rho::Float64; err::Float64
end

all_rows = BenchRow[]

# --- Complexity slope helper ---
function log_slope(xs, ys)
    valid = (xs .> 0) .& (ys .> 1e-15)
    lx = log10.(xs[valid]);  ly = log10.(ys[valid])
    length(lx) < 3 && return NaN
    n = length(lx);  i0 = max(1, n÷2)
    lx = lx[i0:end];  ly = ly[i0:end]
    slopes = diff(ly) ./ diff(lx)
    return round(sum(slopes)/length(slopes), digits=2)
end

colors  = [:blue, :red, :green]
markers = [:circle, :rect, :diamond]

function _update_plot(rows, orders, png_path)
    p1 = plot(title="Time vs p  (MF-SSDM, Mathieu d=2)",
              xlabel="p  (steps per period)", ylabel="CPU time (s)",
              xscale=:log10, yscale=:log10, legend=:topleft, grid=:both, minorgrid=true)
    p2 = plot(title="Spectral-radius error vs p",
              xlabel="p", ylabel="|ρ - ρ_ref|",
              xscale=:log10, yscale=:log10, legend=:topright, grid=:both, minorgrid=true)
    p3 = plot(title="Operator dimension D vs p",
              xlabel="p", ylabel="D  (state-vector size)",
              xscale=:log10, yscale=:log10, legend=:topleft, grid=:both, minorgrid=true)
    p4 = plot(title="Work-precision  (Error vs CPU time)",
              xlabel="CPU time (s)", ylabel="|ρ - ρ_ref|",
              xscale=:log10, yscale=:log10, legend=:topright, grid=:both, minorgrid=true)

    for (ci, order) in enumerate(orders)
        ord_rows = filter(r -> r.order == order, rows)
        isempty(ord_rows) && continue
        pv = [r.p      for r in ord_rows]
        tv = [r.time_s for r in ord_rows]
        Dv = [r.D      for r in ord_rows]
        ev = [r.err    for r in ord_rows]
        st = log_slope(pv, tv)
        se = log_slope(pv, ev)
        sD = log_slope(pv, Dv)
        plot!(p1, pv, tv, marker=markers[ci], color=colors[ci], label="Order $order (slope≈$st)")
        plot!(p2, pv, ev, marker=markers[ci], color=colors[ci], label="Order $order (slope≈$se)")
        plot!(p3, pv, Dv, marker=markers[ci], color=colors[ci], label="Order $order (slope≈$sD)")
        plot!(p4, tv, ev, marker=markers[ci], color=colors[ci], label="Order $order")
    end

    # Reference O(p²) / O(p³) lines anchored to order-0 midpoint
    ord0 = filter(r -> r.order == 0, rows)
    if length(ord0) >= 2
        pv = [r.p for r in ord0];  tv = [r.time_s for r in ord0]
        i0 = max(1, length(pv)÷2)
        p0 = Float64(pv[i0]);  t0 = tv[i0]
        pr = Float64.([pv[1], pv[end]])
        plot!(p1, pr, t0 .* (pr./p0).^2, ls=:dash, color=:gray,  lw=2, label="O(p²)")
        plot!(p1, pr, t0 .* (pr./p0).^3, ls=:dot,  color=:black, lw=2, label="O(p³)")
    end

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1300,1000),
               plot_title="MF-SSDM CPU Complexity — Stochastic Mathieu (d=2, τ=1)")
    savefig(fig, png_path)
end

# --- Output file paths (fixed at start so CSV/PNG have one consistent name) ---
ts_stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
csv_path = "benchmark_mf_complexity_$(ts_stamp).csv"
png_path = "benchmark_mf_complexity_$(ts_stamp).png"
println("Output files:")
println("  CSV → $csv_path")
println("  PNG → $png_path\n")

for order in orders
    println("=== SemiDiscretization order $order ===")
    for p in ps_all
        method = SemiDiscretization(order, P/p)
        rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
        dm     = DiscreteMapping_M2_MF(rst)

        d = 2; r = div(rst.n, d) - 1
        D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]

        # warm-up
        ρ = spectralRadiusOfMapping_MF(dm)
        err = abs(ρ - ρ_ref)

        bm = @benchmark spectralRadiusOfMapping_MF($dm) samples=3 evals=1 seconds=1.0
        t  = median(bm).time / 1e9
        m  = bm.memory / 1024^2

        @printf("  p=%4d  D=%7d  t=%8.4f s  mem=%7.1f MB  ρ=%.8f  err=%.2e\n",
                p, D, t, m, ρ, err)

        push!(all_rows, BenchRow(order, p, D, t, m, ρ, err))

        # --- Update CSV after every data point ---
        open(csv_path, "w") do io
            println(io, "order,p,D,time_s,mem_MB,rho,err")
            for r in all_rows
                @printf(io, "%d,%d,%d,%.6f,%.4f,%.10f,%.6e\n",
                        r.order, r.p, r.D, r.time_s, r.mem_MB, r.rho, r.err)
            end
        end

        # --- Update PNG after every data point ---
        _update_plot(all_rows, orders, png_path)

        if t > TIME_LIMIT
            println("  → time limit reached, skipping larger p for order $order")
            break
        end
    end
    println()
end
println("Done. CSV → $csv_path")
println("      PNG → $png_path")
