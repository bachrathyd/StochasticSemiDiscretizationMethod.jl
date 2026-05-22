using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra, StaticArrays, SparseArrays, CUDA
using Plots, Printf, DelimitedFiles, Dates

BLAS.set_num_threads(1)

# ── Problem definition ────────────────────────────────────────────────────────
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

println("GPU: ", CUDA.name(CUDA.device()))
println()

# ── Reference spectral radius (high-accuracy) ─────────────────────────────────
println("Computing reference ρ (order=2, p=500, MF)...")
ref_method = SemiDiscretization(2, P/500)
ref_rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, ref_method, τ)
ref_dm     = DiscreteMapping_M2_MF(ref_rst)
ρ_ref      = spectralRadiusOfMapping_MF(ref_dm)
@printf("  ρ_ref = %.10f\n\n", ρ_ref)

# ── JIT warm-up (p=20 for all three methods) ──────────────────────────────────
println("JIT warm-up...")
_m  = SemiDiscretization(1, P/20)
_r  = StochasticSemiDiscretizationMethod.calculateResults(lddep, _m, τ)
_d  = DiscreteMapping_M2_MF(_r)
_d2 = DiscreteMapping_M2(_r)   # original M2 (non-MF)

spectralRadiusOfMapping(_d2)             # original (matrix-based)
spectralRadiusOfMapping_MF(_d)           # MF
spectralRadiusOfMapping_GPU_v3(_d); CUDA.synchronize()   # GPU v3
println("done\n")

# ── p grid: ~50 points log-spaced 10…5000 ────────────────────────────────────
ps_raw = unique(round.(Int, exp10.(range(log10(10), log10(5000), length=50))))
TIME_LIMIT = 10.0

# We will stop each method independently once it exceeds the limit.
struct Row
    method::String
    order::Int
    p::Int
    D::Int
    t_s::Float64      # wall time (s)
    mem_MB::Float64   # allocated memory (MB)
    rho::Float64
    err::Float64
end

rows = Row[]

# Helper: state-space dimension for M2 MF mapping
function D_of(rst, d)
    r = div(rst.n, d) - 1
    StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
end

# ── Benchmark loop ────────────────────────────────────────────────────────────
for order in [0, 1, 2]
    println("="^70)
    println("Order $order")
    println("="^70)

    # ── Method A: Original (matrix DiscreteMapping_M2) ──
    println("  [Original]")
    orig_stopped = false
    for p in ps_raw
        orig_stopped && break
        method = SemiDiscretization(order, P/p)
        rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
        dm     = DiscreteMapping_M2(rst)
        d      = 2   # problem dimension

        # quick probe
        t_probe = @elapsed spectralRadiusOfMapping(dm)
        if t_probe > TIME_LIMIT
            orig_stopped = true
            @printf("    p=%4d  STOPPED (probe %.1f s)\n", p, t_probe)
            break
        end

        # single or repeated measurement depending on probe time
        if t_probe < 0.5
            # repeat until we accumulate ≥1 s of measurement
            n_rep = max(1, ceil(Int, 1.0 / t_probe))
            n_rep = min(n_rep, 20)
            t_total = @elapsed for _ in 1:n_rep; spectralRadiusOfMapping(dm); end
            t_s = t_total / n_rep
        else
            t_s = t_probe   # single shot is enough
        end
        mem_bytes = @allocated spectralRadiusOfMapping(dm)
        rho       = spectralRadiusOfMapping(dm)
        err       = abs(rho - ρ_ref) / ρ_ref

        # D for original: full D×D matrix; use rst.n as proxy
        D_orig = rst.n^2   # covariance matrix dimension (n×n)

        @printf("    p=%4d  D=%7d  t=%.4f s  mem=%.1f MB  err=%.2e\n",
                p, D_orig, t_s, mem_bytes/1024^2, err)
        push!(rows, Row("orig", order, p, D_orig, t_s, mem_bytes/1024^2, rho, err))

        if t_s > TIME_LIMIT
            orig_stopped = true
            println("    Time limit reached.")
            break
        end
    end

    # ── Method B: Multiplication-Free (CPU) ──
    println("  [MF-CPU]")
    mf_stopped = false
    for p in ps_raw
        mf_stopped && break
        method = SemiDiscretization(order, P/p)
        rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
        dm     = DiscreteMapping_M2_MF(rst)
        d      = size(dm.coeffs.det[1][1][1], 1)
        D      = D_of(rst, d)

        t_probe = @elapsed spectralRadiusOfMapping_MF(dm)
        if t_probe > TIME_LIMIT
            mf_stopped = true
            @printf("    p=%4d  STOPPED (probe %.1f s)\n", p, t_probe)
            break
        end

        if t_probe < 0.5
            n_rep = max(1, min(20, ceil(Int, 1.0 / t_probe)))
            t_total = @elapsed for _ in 1:n_rep; spectralRadiusOfMapping_MF(dm); end
            t_s = t_total / n_rep
        else
            t_s = t_probe
        end
        mem_bytes = @allocated spectralRadiusOfMapping_MF(dm)
        rho       = spectralRadiusOfMapping_MF(dm)
        err       = abs(rho - ρ_ref) / ρ_ref

        @printf("    p=%4d  D=%7d  t=%.4f s  mem=%.1f MB  err=%.2e\n",
                p, D, t_s, mem_bytes/1024^2, err)
        push!(rows, Row("mf", order, p, D, t_s, mem_bytes/1024^2, rho, err))

        if t_s > TIME_LIMIT
            mf_stopped = true
            println("    Time limit reached.")
            break
        end
    end

    # ── Method C: GPU v3 (cooperative) ──
    println("  [GPU-v3]")
    gpu_stopped = false
    for p in ps_raw
        gpu_stopped && break
        method = SemiDiscretization(order, P/p)
        rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
        dm     = DiscreteMapping_M2_MF(rst)
        d      = size(dm.coeffs.det[1][1][1], 1)
        D      = D_of(rst, d)

        t_probe = @elapsed begin
            spectralRadiusOfMapping_GPU_v3(dm); CUDA.synchronize()
        end
        if t_probe > TIME_LIMIT
            gpu_stopped = true
            @printf("    p=%4d  STOPPED (probe %.1f s)\n", p, t_probe)
            break
        end

        if t_probe < 0.5
            n_rep = max(1, min(5, ceil(Int, 1.0 / t_probe)))
            t_total = @elapsed for _ in 1:n_rep
                spectralRadiusOfMapping_GPU_v3(dm); CUDA.synchronize()
            end
            t_s = t_total / n_rep
        else
            t_s = t_probe
        end
        mem_bytes = @allocated spectralRadiusOfMapping_GPU_v3(dm)
        rho       = spectralRadiusOfMapping_GPU_v3(dm); CUDA.synchronize()
        err       = abs(rho - ρ_ref) / ρ_ref

        @printf("    p=%4d  D=%7d  t=%.4f s  mem=%.1f MB  err=%.2e\n",
                p, D, t_s, mem_bytes/1024^2, err)
        push!(rows, Row("gpu3", order, p, D, t_s, mem_bytes/1024^2, rho, err))

        if t_s > TIME_LIMIT
            gpu_stopped = true
            println("    Time limit reached.")
            break
        end
    end
end

# ── Save CSV ──────────────────────────────────────────────────────────────────
timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
csv_path  = "work_precision_full_$(timestamp).csv"
open(csv_path, "w") do io
    println(io, "method,order,p,D,t_s,mem_MB,rho,err")
    for r in rows
        @printf(io, "%s,%d,%d,%d,%.6f,%.4f,%.10f,%.6e\n",
                r.method, r.order, r.p, r.D, r.t_s, r.mem_MB, r.rho, r.err)
    end
end
println("\nCSV saved: $csv_path")

# ── Plot ──────────────────────────────────────────────────────────────────────
method_styles = Dict(
    "orig" => (label_prefix="Orig",   color=:red,    ls=:solid,  marker=:square),
    "mf"   => (label_prefix="MF-CPU", color=:blue,   ls=:solid,  marker=:circle),
    "gpu3" => (label_prefix="GPU-v3", color=:green,  ls=:solid,  marker=:diamond),
)
order_alpha = Dict(0 => 1.0, 1 => 0.7, 2 => 0.4)
order_dash  = Dict(0 => :solid, 1 => :dash, 2 => :dot)

p1 = plot(title="Time vs p",    xscale=:log10, yscale=:log10,
          xlabel="p", ylabel="Wall time (s)", legend=:topleft)
p2 = plot(title="Error vs p",   xscale=:log10, yscale=:log10,
          xlabel="p", ylabel="|ρ−ρ_ref|/ρ_ref", legend=:topright)
p3 = plot(title="Memory vs p",  xscale=:log10, yscale=:log10,
          xlabel="p", ylabel="Allocated memory (MB)", legend=:topleft)
p4 = plot(title="Work-precision (time vs error)", xscale=:log10, yscale=:log10,
          xlabel="Wall time (s)", ylabel="|ρ−ρ_ref|/ρ_ref", legend=:topright)

for mkey in ["orig", "mf", "gpu3"]
    sty = method_styles[mkey]
    for order in [0, 1, 2]
        sub = filter(r -> r.method == mkey && r.order == order, rows)
        isempty(sub) && continue
        pv  = [r.p     for r in sub]
        tv  = [r.t_s   for r in sub]
        ev  = [max(r.err, 1e-14) for r in sub]   # floor for log scale
        mv  = [r.mem_MB for r in sub]

        lbl = "$(sty.label_prefix) ord$order"
        ls  = order_dash[order]

        plot!(p1, Float64.(pv), tv, label=lbl, color=sty.color, ls=ls,
              marker=sty.marker, ms=4, lw=1.5)
        plot!(p2, Float64.(pv), ev, label=lbl, color=sty.color, ls=ls,
              marker=sty.marker, ms=4, lw=1.5)
        plot!(p3, Float64.(pv), mv, label=lbl, color=sty.color, ls=ls,
              marker=sty.marker, ms=4, lw=1.5)
        plot!(p4, tv, ev, label=lbl, color=sty.color, ls=ls,
              marker=sty.marker, ms=4, lw=1.5)
    end
end

# reference lines: O(p^-1), O(p^-2) on error plot
p_ref = [10.0, 5000.0]
e0    = filter(r -> r.method=="mf" && r.order==1 && r.p==ps_raw[1], rows)
if !isempty(e0)
    e_start = e0[1].err
    plot!(p2, p_ref, e_start .* (p_ref ./ ps_raw[1]).^(-1),
          ls=:dashdot, color=:gray, lw=1, label="O(p⁻¹)")
    plot!(p2, p_ref, e_start .* (p_ref ./ ps_raw[1]).^(-2),
          ls=:dot,    color=:black, lw=1, label="O(p⁻²)")
end

hline!(p1, [TIME_LIMIT], ls=:dash, color=:black, lw=1.5, label="10 s limit")

fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1500, 1100),
           plot_title="Stochastic Delayed Mathieu — Orig vs MF-CPU vs GPU-v3  [$(CUDA.name(CUDA.device()))]",
           margin=8Plots.mm)

png_path = "work_precision_full_$(timestamp).png"
savefig(fig, png_path)
println("PNG  saved: $png_path")
println("Done.")
