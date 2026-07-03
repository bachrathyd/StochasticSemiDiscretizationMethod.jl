# Final render of the classical-vs-MF work-precision figure from wp_ultra.csv.
# Drops the first (JIT-compilation-polluted) point of each method.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, DelimitedFiles, Printf

raw,_ = readdlm(joinpath(@__DIR__,"wp_ultra.csv"), ','; header=true)
meth=String.(raw[:,1]); p=Int.(raw[:,2]); t=Float64.(raw[:,4]); mem=Float64.(raw[:,5]); err=Float64.(raw[:,6])
sel(m) = findall((meth.==m) .& (p .> 8))
cl=sel("classical"); mf=sel("MF")

kw = (framestyle=:box, guidefontsize=11, tickfontsize=9, legendfontsize=9,
      xscale=:log10, minorgrid=true, gridalpha=0.25, minorgridalpha=0.10)
p1 = plot(; xlabel="", ylabel="wall-clock time [s]", yscale=:log10,
          title="(a) cost vs resolution", titleloc=:left, titlefontsize=11, legend=:topleft, kw...)
p2 = plot(; xlabel="", ylabel="allocated memory [MB]", yscale=:log10,
          title="(b) memory vs resolution", titleloc=:left, titlefontsize=11, legend=false, kw...)
p3 = plot(; xlabel="p (steps per period)", ylabel="|ρ − ρ_ref|", yscale=:log10,
          title="(c) error vs resolution", titleloc=:left, titlefontsize=11, legend=false, kw...)
p4 = plot(; xlabel="wall-clock time [s]", ylabel="|ρ − ρ_ref|", yscale=:log10,
          title="(d) error vs cost", titleloc=:left, titlefontsize=11, legend=false, kw...)
for (ii,c,mk,lab) in ((cl,:black,:circle,"classical (explicit period product)"),
                      (mf,:crimson,:utriangle,"multiplication-free"))
    plot!(p1, p[ii], t[ii], color=c, marker=mk, ms=4, lw=1.8, label=lab)
    plot!(p2, p[ii], mem[ii], color=c, marker=mk, ms=4, lw=1.8, label="")
    plot!(p3, p[ii], err[ii], color=c, marker=mk, ms=4, lw=1.8, label="")
    plot!(p4, t[ii], err[ii], color=c, marker=mk, ms=4, lw=1.8, label="")
end
plot!(p1, [12,192.0], t[mf[1]] .* ([12,192.0] ./ 12).^2 .* 0.6, ls=:dot, color=:gray60, label="∝ p²")
plot!(p1, [12,192.0], t[cl[1]] .* ([12,192.0] ./ 12).^4 .* 2.0, ls=:dot, color=:gray30, label="∝ p⁴")
plt = plot(p1,p2,p3,p4, layout=(2,2), size=(1150,860), dpi=300,
           left_margin=6Plots.mm, bottom_margin=5Plots.mm)
savefig(plt, joinpath(@__DIR__,"wp_ultra.png")); savefig(plt, joinpath(@__DIR__,"wp_ultra.pdf"))
dst = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"
for f in ("wp_ultra.png","wp_ultra.pdf"); cp(joinpath(@__DIR__,f), joinpath(dst,f); force=true); end
# ratios for the paper text
i1=findfirst(==(192), p[cl]); j1=findfirst(==(192), p[mf])
@printf("at p=192: time ratio %.0fx, memory ratio %.0fx\n",
        t[cl][i1]/t[mf][j1], mem[cl][i1]/mem[mf][j1])
println("done")
