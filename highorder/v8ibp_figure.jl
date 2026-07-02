# Paper figure: convergence of ρ(H) for the HARD PD-Mathieu.
# Curves: v7 (sampling-based, expected collapse), v8-direct (integrated-history),
# v8-IBP. v8/IBP errors are from the certified run (ref = 1.324866438112,
# |ref−arbiter| = 9.4e-13); v7 computed here.
include(joinpath(@__DIR__, "cov_colloc_v8_ibp.jl"))
using Printf, Plots

Afun(t)=[0.0 1.0; -(1.0+0.8cos(2π*t)) -0.1]
Bfun(t)=[0.0 0.0; 0.40*(1+0.3cos(2π*t)) 0.45*(1+0.4cos(2π*t))]
αfun(t)=[0.0 0.0; 0.5 0.0]
βfun(t)=[0.0 0.0; 0.35 0.0]
pb = Prob(2,1.0,1.0, Afun, Bfun, αfun, βfun)
const ρref = 1.324866438112

ps = [4,6,8,12,16,24]
errs_v7 = Float64[]
for p in ps
    ρ = rho_H_krylov(build_v7(pb,3,p); offdiag=:causal)
    push!(errs_v7, abs(ρ-ρref))
    @printf("v7 GL3 p=%2d err=%.2e\n", p, errs_v7[end]); flush(stdout)
end
rates=[log(errs_v7[i]/errs_v7[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
println("v7 GL3 rates: ", join([@sprintf("%.2f",r) for r in rates], " "))

# certified data from v8ibp_hard run
errs_v8  = [1.05e-06, 9.01e-08, 1.61e-08, 1.42e-09, 2.48e-10, 7.76e-12]
errs_ibp = [4.38e-07, 3.33e-08, 5.70e-09, 5.06e-10, 9.11e-11, 2.21e-11]

plt = plot(xlabel="p  (steps per period)", ylabel="|ρ(H) − ρ_ref|",
           xscale=:log10, yscale=:log10, legend=:bottomleft,
           size=(700,520), framestyle=:box, minorgrid=false)
plot!(plt, ps, errs_v7,  marker=:circle,    color=:firebrick,  label="sampling-based (v7-type), GL3")
plot!(plt, ps, errs_v8,  marker=:utriangle, color=:seagreen,   label="integrated-history (v8), GL3")
plot!(plt, ps, errs_ibp, marker=:diamond,   color=:royalblue,  label="v8 + IBP, GL3")
# slope guides
plot!(plt, ps, 2e-4 .* (ps ./ 4.0) .^ -2, ls=:dash, color=:gray, label="O(h²), O(h⁶) guides")
plot!(plt, ps, 1.0e-6 .* (ps ./ 4.0) .^ -6, ls=:dash, color=:gray, label="")
out = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images\pd_mathieu_orders.png"
savefig(plt, out)
savefig(plt, joinpath(@__DIR__, "out_pd_orders.png"))
println("figure written: $out")
