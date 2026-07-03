# Journal figure (restyled): the rough-read order collapse and its two remedies,
# on the hard PD-Mathieu at S=3 (nominal order six).
#   * sampling-based treatment (v7 engine)  → collapses to O(h²)
#   * integrated-history treatment (v8)     → full nominal order 6
#   * v8 + IBP                              → same order, smaller constants
# v7 is computed here (cheap); v8/IBP GL3 errors are read from the certified
# dense run out_grand_orders_pub.csv. Styling matches the other journal figures.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra, Printf, Plots, DelimitedFiles
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v8_ibp.jl"))

Afun(t)=[0.0 1.0; -(1.0+0.8cos(2π*t)) -0.1]
Bfun(t)=[0.0 0.0; 0.40*(1+0.3cos(2π*t)) 0.45*(1+0.4cos(2π*t))]
αfun(t)=[0.0 0.0; 0.5 0.0]
βfun(t)=[0.0 0.0; 0.35 0.0]
pb = Prob(2,1.0,1.0, Afun, Bfun, αfun, βfun)
const ρref = 1.324866438112

ps7 = [4,6,8,12,16,24,32,48,64]
errs_v7 = Float64[]
for p in ps7
    ρ = rho_H_krylov(build_v7(pb,3,p); offdiag=:causal)
    push!(errs_v7, abs(ρ-ρref))
    @printf("v7 GL3 p=%2d err=%.2e\n", p, errs_v7[end]); flush(stdout)
end

raw, _ = readdlm(joinpath(@__DIR__,"out_grand_orders_pub.csv"), ','; header=true)
names=String.(raw[:,1]); psA=Int.(raw[:,2]); errA=Float64.(raw[:,3])
sel8  = names.=="v8 GL3";  ps8=psA[sel8];  errs8=errA[sel8]
selI  = names.=="IBP GL3"; psI=psA[selI];  errsI=errA[selI]
keep8 = errs8 .> 2e-11; keepI = errsI .> 2e-11        # cut the floor tail here

plt = plot(xlabel="p  (steps per principal period)", ylabel="|ρ − ρ_ref|",
           xscale=:log10, yscale=:log10, legend=:bottomleft,
           size=(900,620), framestyle=:box, dpi=300,
           guidefontsize=13, tickfontsize=11, legendfontsize=10,
           left_margin=6Plots.mm, bottom_margin=5Plots.mm,
           minorgrid=true, gridalpha=0.25, minorgridalpha=0.10,
           xticks=10.0 .^ (0:2), yticks=10.0 .^ (-12:2:0))
for (ord, anch, px) in ((2,3e-4,4.0),(6,1.05e-6,4.0))
    pg=10 .^ range(log10(4.0), log10(80.0), length=2)
    plot!(plt, pg, anch .* (px ./ pg).^ord, color=:gray70, ls=:dot, lw=1, label="")
    annotate!(plt, pg[2]*0.9, anch*(px/pg[2])^ord*3.5,
              Plots.text("p⁻$ord", 9, :gray50, :right))
end
plot!(plt, ps7, errs_v7, marker=:circle, ms=5, markerstrokewidth=0.4, lw=1.8,
      color=:firebrick, label="sampling-based read (order collapse)")
plot!(plt, ps8[keep8], errs8[keep8], marker=:utriangle, ms=5, markerstrokewidth=0.4,
      lw=1.8, color=:seagreen, label="integrated-history read")
plot!(plt, psI[keepI], errsI[keepI], marker=:diamond, ms=5, markerstrokewidth=0.4,
      lw=1.8, ls=:dash, color=:royalblue, label="integrated-history + IBP")
savefig(plt, joinpath(@__DIR__,"out_pd_orders.png"))
savefig(plt, joinpath(@__DIR__,"out_pd_orders.pdf"))
dst = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"
cp(joinpath(@__DIR__,"out_pd_orders.png"), joinpath(dst,"pd_mathieu_orders.png"); force=true)
cp(joinpath(@__DIR__,"out_pd_orders.pdf"), joinpath(dst,"pd_mathieu_orders.pdf"); force=true)
println("done — pd_mathieu_orders restyled")
