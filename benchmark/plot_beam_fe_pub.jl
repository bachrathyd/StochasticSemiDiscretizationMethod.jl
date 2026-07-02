# Journal figure for the FE-beam modal-convergence study.
# Re-plots benchmark/beam_fe.csv (no recomputation): two panels —
# (a) ρ(H) vs number of retained modes, (b) |ρ − ρ_converged| vs n_m (semilog)
# with the covariance problem size D annotated.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, Printf, DelimitedFiles

raw, hdr = readdlm(joinpath(@__DIR__, "beam_fe.csv"), ','; header=true)
nm  = Int.(raw[:,1]); d = Int.(raw[:,2]); D = Int.(raw[:,3])
ρ   = Float64.(raw[:,4]); t = Float64.(raw[:,5])
ρc  = ρ[end]

pa = plot(nm, ρ, marker=:circle, ms=5, lw=1.8, color=:dodgerblue, label="",
          xlabel="number of retained modes  n_m", ylabel="ρ(H)",
          framestyle=:box, guidefontsize=12, tickfontsize=10,
          title="(a)", titleloc=:left, titlefontsize=12)
hline!(pa, [ρc], color=:gray, ls=:dash, label="")
for i in eachindex(nm)
    annotate!(pa, nm[i], ρ[i]-6e-5,
              Plots.text(@sprintf("D=%.2g", float(D[i])), 7, :gray40, :left, rotation=-30))
end

err = abs.(ρ .- ρc); err[end] = NaN   # last point is the reference itself
pb = plot(nm[1:end-1], max.(err[1:end-1],1e-9), marker=:square, ms=5, lw=1.8,
          color=:crimson, yscale=:log10, label="",
          xlabel="number of retained modes  n_m", ylabel="|ρ − ρ(n_m=16)|",
          framestyle=:box, guidefontsize=12, tickfontsize=10,
          title="(b)", titleloc=:left, titlefontsize=12)

plt = plot(pa, pb, layout=(1,2), size=(1150,430), dpi=300,
           left_margin=7Plots.mm, bottom_margin=7Plots.mm)
savefig(plt, joinpath(@__DIR__, "beam_fe_pub.png"))
savefig(plt, joinpath(@__DIR__, "beam_fe_pub.pdf"))
dst = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"
for f in ("beam_fe_pub.png","beam_fe_pub.pdf")
    cp(joinpath(@__DIR__,f), joinpath(dst, replace(f,"beam_fe_pub"=>"beam_mesh_convergence")); force=true)
end
println("done — beam_fe_pub.png/pdf (+ paper images/beam_mesh_convergence)")
