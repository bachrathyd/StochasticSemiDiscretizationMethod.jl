# Final journal rendering of the grand order diagram.
# Re-plots out_grand_orders_pub.csv (no recomputation). Each curve is truncated
# two points after it first drops below the eigensolver accuracy floor — the
# raw wandering-on-the-floor tail stays in the CSV but clutters the figure.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, Printf, DelimitedFiles

const FLOOR = 2e-11      # observed Krylov accuracy plateau
const CSV = joinpath(@__DIR__, "out_grand_orders_pub.csv")
const PNG = joinpath(@__DIR__, "out_grand_orders_pub.png")
const PDF = joinpath(@__DIR__, "out_grand_orders_pub.pdf")
const PAPER_IMG = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"

raw, _ = readdlm(CSV, ','; header=true)
names = String.(raw[:,1]); ps = Int.(raw[:,2]); errs = Float64.(raw[:,3])

order  = ["SDM q=2","SDM q=4","v8 GL1","IBP GL1","v8 GL2","IBP GL2",
          "v8 GL3","IBP GL3","v8 GL4","IBP GL4","v8 GL5","IBP GL5"]
scol   = Dict(1=>:dodgerblue, 2=>:seagreen, 3=>:purple, 4=>:darkorange, 5=>:crimson)
colof(n) = n=="SDM q=2" ? :black : n=="SDM q=4" ? :gray40 :
           scol[parse(Int, n[end])]

plt = plot(xlabel="p  (steps per principal period)",
           ylabel="|ρ − ρ_ref|",
           xscale=:log10, yscale=:log10, legend=:outerright,
           size=(1200,760), framestyle=:box, dpi=300,
           guidefontsize=13, tickfontsize=11, legendfontsize=10,
           left_margin=8Plots.mm, bottom_margin=5Plots.mm,
           minorgrid=true, gridalpha=0.25, minorgridalpha=0.10,
           xticks=10.0 .^ (0:4), yticks=10.0 .^ (-12:2:0),
           ylims=(3e-13, 1e-1), xlims=(3, 3000))
for (ord, anch, px) in ((1,2e-2,300.0),(2,2e-3,80.0),(4,2e-5,30.0),
                        (6,2e-7,16.0),(8,2e-9,10.0))
    pg = 10 .^ range(log10(4.0), log10(2000.0), length=2)
    eg = anch .* (px ./ pg).^ord
    keep = eg .> 4e-13
    plot!(plt, pg[keep], eg[keep], color=:gray70, ls=:dot, lw=1, label="")
    # label where the guide crosses y=1e-11 (or at the right edge if it doesn't)
    plab = min(px*(anch/1e-11)^(1/ord), 1400.0)
    elab = anch*(px/plab)^ord
    annotate!(plt, plab*1.12, elab*3.0, Plots.text("p⁻$ord", 9, :gray50, :left))
end
hline!(plt, [FLOOR], color=:gray, ls=:dashdot, lw=1, label="solver floor")
for name in order
    idx = findall(==(name), names)
    isempty(idx) && continue
    p = ps[idx]; e = errs[idx]
    # truncate two points after first drop below the floor
    ic = findfirst(<(FLOOR), e)
    if ic !== nothing
        p = p[1:min(end, ic+1)]; e = e[1:min(end, ic+1)]
    end
    ls = startswith(name,"IBP") ? :dash : :solid
    mk = startswith(name,"SDM") ? :circle : (startswith(name,"IBP") ? :diamond : :utriangle)
    plot!(plt, p, max.(e,4e-13), marker=mk, markersize=4.5, markerstrokewidth=0.4,
          lw=1.8, ls=ls, color=colof(name), label=name)
end
savefig(plt, PNG); savefig(plt, PDF)
for f in (basename(PNG), basename(PDF), basename(CSV))
    try cp(joinpath(@__DIR__,f), joinpath(PAPER_IMG, replace(f,"out_"=>"")); force=true)
    catch e; @warn "copy failed" f e end
end
println("done — $PNG / $PDF (+ paper images)")
