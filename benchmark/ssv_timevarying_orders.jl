# Convergence orders of the time-varying-delay (vT) collocation engine on the
# SSV turning model — the delay τ(t) = 2π/(Ω₀(1+RVA·sin(RVF·t))) is genuinely
# time-varying, so the aligned 2S engine does not apply; this measures what the
# fractional-limit integrated-history engine actually delivers (floor S+1,
# observed ≈ 2S) against the classical MF-factored path (order 1).
#
# Incremental protocol: results are appended to CSV and the figure re-rendered
# after EVERY solve, so partial runs are usable. Outputs:
#   benchmark/ssv_vt_orders.csv
#   assets/TimeVaryingDelayConvergence.png            (README)
#   journal_paper/images/ssv_vt_orders.{png,pdf}      (paper; skipped if absent)
using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
BLAS.set_num_threads(1)
using StochasticSemiDiscretizationMethod, StaticArrays, Plots, Printf, DelimitedFiles

const CSV = joinpath(@__DIR__, "ssv_vt_orders.csv")
const ASSET = joinpath(@__DIR__, "..", "assets", "TimeVaryingDelayConvergence.png")
const PAPER_IMG = joinpath(@__DIR__, "..", "journal_paper", "images")

# ---- SSV turning model (temo/temporal study parameters) ---------------------
Ω0 = 0.87; RVA = 0.1; RVF = 0.1; ζ = 0.05; w = 0.4; σ = 0.1
T = 2π / RVF
τf(t) = (2π) / (Ω0 * (1.0 + RVA * sin(RVF * t)))
Af(t) = @SMatrix [0.0 1.0; -(1.0+w) -2ζ]
Bf(t) = @SMatrix [0.0 0.0; w 0.0]
z2 = @SMatrix zeros(2, 2)
prob = LDDEProblem(ProportionalMX(Af), [DelayMX(τf, Bf)],
                   [stCoeffMX(1, ProportionalMX(t -> z2))],
                   [stCoeffMX(1, DelayMX(τf, t -> z2))],
                   Additive(2), [stAdditive(1, Additive(@SVector [0.0, σ]))], 1)

# ---- incremental storage ----------------------------------------------------
have = Dict{Tuple{String,Int},NTuple{2,Float64}}()
if isfile(CSV)
    raw, _ = readdlm(CSV, ','; header=true)
    for k in axes(raw, 1)
        have[(String(raw[k,1]), Int(raw[k,2]))] = (Float64(raw[k,3]), Float64(raw[k,4]))
    end
    println("resuming: $(length(have)) cached points")
end
function save_all()
    open(CSV, "w") do io
        println(io, "case,p,rho,var")
        for ((c, p), (ρ, v)) in sort(collect(have); by=x->(x[1][1], x[1][2]))
            println(io, "$c,$p,$ρ,$v")
        end
    end
end

function solve!(case, p; S=0, q=2)
    haskey(have, (case, p)) && return have[(case, p)]
    t0 = time()
    ρ, v = if case == "classical"
        (spectralRadiusOfMoment(prob, T, p; method=ClassicalSD(q)),
         stationaryVariance(prob, T, p; method=ClassicalSD(q)))
    else
        (spectralRadiusOfMoment(prob, T, p; method=Collocation(S), verbosity=0),
         stationaryVariance(prob, T, p; method=Collocation(S), verbosity=0))
    end
    have[(case, p)] = (ρ, v)
    save_all()
    @printf("  %-10s p=%4d  ρ=%.12f  var=%.12f  (%.1f s)\n", case, p, ρ, v, time()-t0)
    flush(stdout)
    (ρ, v)
end

# ---- reference: Richardson from the finest S=3 ratio-2 triple ---------------
function richardson(vals)               # fitted-order extrapolation, last triple
    q2 = (vals[2] - vals[1]) / (vals[3] - vals[2])
    vals[3] + (vals[3] - vals[2]) / (q2 - 1)
end

function render()
    ps3 = [60, 120, 240]
    all(p -> haskey(have, ("S3", p)), ps3) || return   # need the reference first
    ρref = richardson([have[("S3", p)][1] for p in ps3])
    vref = richardson([have[("S3", p)][2] for p in ps3])
    plt = plot(layout=(1, 2), size=(1200, 460), dpi=300, framestyle=:box,
               legend=:bottomleft, guidefontsize=12, tickfontsize=10,
               left_margin=6Plots.mm, bottom_margin=6Plots.mm)
    series = [("classical", "classical SD (q=2), order 1", :black, :circle),
              ("S1", "GL collocation S=1", :seagreen, :utriangle),
              ("S2", "GL collocation S=2", :royalblue, :diamond),
              ("S3", "GL collocation S=3", :firebrick, :square)]
    for (panel, ref, lab) in ((1, ρref, "error in ρ(H)"), (2, vref, "error in Var(x)"))
        for (case, name, col, mk) in series
            pts = sort([(p, v) for ((c, p), v) in have if c == case])
            isempty(pts) && continue
            xs = [p for (p, _) in pts]
            es = [abs(v[panel] - ref) for (_, v) in pts]
            keep = es .> 0
            plot!(plt[panel], xs[keep], es[keep]; xscale=:log10, yscale=:log10,
                  marker=mk, color=col, label=(panel == 1 ? name : ""))
        end
        plot!(plt[panel]; xlabel="steps per period  p", ylabel=lab,
              title=(panel == 1 ? "SSV turning — time-varying delay τ(t)" : ""))
    end
    # slope guides on the ρ panel
    for (ord, x0, y0) in ((1, 600.0, 2e-3), (4, 90.0, 3e-5), (6, 90.0, 3e-7))
        xs = [x0, 2x0]
        plot!(plt[1], xs, y0 .* (x0 ./ xs) .^ ord; color=:gray, ls=:dash,
              label="", annotations=(2.2x0, y0 * 2.0^-ord, Plots.text("$ord", 9, :gray)))
    end
    png(plt, ASSET)
    if isdir(PAPER_IMG)
        try
            png(plt, joinpath(PAPER_IMG, "ssv_vt_orders.png"))
            savefig(plt, joinpath(PAPER_IMG, "ssv_vt_orders.pdf"))
        catch e
            @warn "paper image copy failed" e
        end
    end
    println("  figure updated (ρ* = $ρref, var* = $vref)")
end

# ---- ladders (coarse→fine so partial runs already render) -------------------
for p in (60, 120, 240);            solve!("S3", p; S=3); render(); end
for p in (250, 500, 1000, 2000, 4000); solve!("classical", p); render(); end
for p in (40, 80, 160);             solve!("S2", p; S=2); render(); end
for p in (80, 160, 320);            solve!("S1", p; S=1); render(); end
for p in (40, 80, 160);             solve!("S3", p; S=3); render(); end

# measured orders (LSQ over the ladder, vs the Richardson reference)
ρref = richardson([have[("S3", p)][1] for p in (60, 120, 240)])
for (case, ps) in (("S1", [80, 160, 320]), ("S2", [40, 80, 160]), ("S3", [40, 80, 160]))
    es = [abs(have[(case, p)][1] - ρref) for p in ps]
    sl = log2(es[1] / es[end]) / log2(ps[end] / ps[1])
    @printf("%s: ρ errors %s → slope %.2f\n", case, string(es), sl)
end
println("DONE")
