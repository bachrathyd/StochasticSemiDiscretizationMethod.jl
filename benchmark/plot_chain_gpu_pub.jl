# Journal figure (appendix): local-GPU vs CPU scaling of the MF second-moment
# spectral-radius solve across system dimension (oscillator chains, 1..5 DOF)
# and resolution p. Re-plots benchmark/chain_dof.csv (no recomputation).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, Printf, DelimitedFiles

raw,_ = readdlm(joinpath(@__DIR__,"chain_dof.csv"), ','; header=true)
nd=Int.(raw[:,1]); d=Int.(raw[:,2]); p=Int.(raw[:,3]); D=Int.(raw[:,4])
tc=Float64.(raw[:,5]); tg=Float64.(raw[:,6])
ρc=Float64.(raw[:,7]); ρg=Float64.(raw[:,8])
@printf("max |ρ_cpu-ρ_gpu|/ρ = %.2e\n", maximum(abs.(ρc.-ρg)./abs.(ρc)))

cols = Dict(1=>:dodgerblue, 2=>:seagreen, 3=>:purple, 4=>:darkorange, 5=>:crimson)
kw = (framestyle=:box, guidefontsize=12, tickfontsize=10, legendfontsize=9,
      xscale=:log10, minorgrid=true, gridalpha=0.25, minorgridalpha=0.10)
p1 = plot(; xlabel="p (steps per period)", ylabel="wall-clock time [s]",
          yscale=:log10, title="(a) CPU (solid) vs GPU (dashed)", titleloc=:left,
          titlefontsize=12, legend=:topleft, kw...)
p2 = plot(; xlabel="p (steps per period)", ylabel="GPU speedup  t_CPU / t_GPU",
          yscale=:log10, title="(b) speedup grows with d and p", titleloc=:left,
          titlefontsize=12, legend=:bottomright, kw...)
hline!(p2, [1.0], color=:gray, ls=:dashdot, lw=1, label="")
for n in sort(unique(nd))
    s = nd.==n
    tcm = max.(tc[s], 1e-3)                     # sub-ms CPU timings: clock floor
    plot!(p1, p[s], tcm, color=cols[n], marker=:circle, ms=3.5, lw=1.6,
          label="$(n)-DOF (d=$(2n)) CPU")
    plot!(p1, p[s], tg[s], color=cols[n], marker=:utriangle, ms=3.5, lw=1.6,
          ls=:dash, label="$(n)-DOF GPU")
    plot!(p2, p[s], tcm ./ tg[s], color=cols[n], marker=:circle, ms=3.5, lw=1.8,
          label="$(n)-DOF (d=$(2n))")
end
plt = plot(p1, p2, layout=(1,2), size=(1300,520), dpi=300,
           left_margin=6Plots.mm, bottom_margin=6Plots.mm)
savefig(plt, joinpath(@__DIR__,"chain_gpu_pub.png"))
savefig(plt, joinpath(@__DIR__,"chain_gpu_pub.pdf"))
dst = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"
for f in ("chain_gpu_pub.png","chain_gpu_pub.pdf")
    cp(joinpath(@__DIR__,f), joinpath(dst, replace(f,"chain_gpu_pub"=>"gpu_chain_scaling")); force=true)
end
println("done — chain_gpu_pub + images/gpu_chain_scaling")
