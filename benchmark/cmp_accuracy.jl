# Accuracy vs the EXACT discretization-free boundary of Iklodi & Dankowicz
# (arXiv:2607.01374), scalar example τ=1, α=-1.5, β=0. At their exact χ=0
# boundary the true ρ(H)=1, so |ρ_p - 1| is pure discretization error.
# Shows order 2S convergence of our GL-S collocation onto their analytic result.
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod, StaticArrays, Printf, Plots, BenchmarkTools
BLAS.set_num_threads(1)

chi(a,b; τ=1.0, α=-1.5, β=0.0) = begin
    μ = (a^2-b^2) ≥ 0 ? sqrt(a^2-b^2) : sqrt(complex(a^2-b^2))
    real(((α+β)^2+2a+2b)*cosh(μ*τ/2) - ((a+b)/μ)*((α-β)^2+2a-2b)*sinh(μ*τ/2))
end
function bstar(a; lo=1.0, hi=2.99)
    flo=chi(a,lo)
    for _ in 1:80; m=(lo+hi)/2; (flo*chi(a,m)≤0) ? (hi=m) : (lo=m; flo=chi(a,m)); end
    (lo+hi)/2
end
scalar_prob(a,b;τ=1.0,α=-1.5) = LDDEProblem(
    ProportionalMX(t->@SMatrix [a]), [DelayMX(τ, t->@SMatrix [b])],
    [stCoeffMX(1,ProportionalMX(t->@SMatrix [α]))],
    [stCoeffMX(1,DelayMX(τ, t->@SMatrix [0.0]))],
    Additive(1), [stAdditive(1,Additive(@SVector [0.0]))])

const a0 = -3.0; const b0 = bstar(a0)
@printf("exact boundary point: a=%.1f, b*=%.12f (χ=%.1e)\n", a0, b0, chi(a0,b0))
prob = scalar_prob(a0,b0)
cputime(S,p) = (b=@benchmark spectralRadiusOfMapping_collocation($prob,1.0,$p;S=$S) samples=6 seconds=1.0 evals=1; minimum(b.times)/1e9)

const PS = [2,3,4,5,6,8,10,12,16,20,24,32,48,64,96,128]
const scol = Dict(1=>:dodgerblue,2=>:seagreen,3=>:purple,4=>:darkorange,5=>:crimson)
data = Dict{Int,NamedTuple}()
for S in 1:5
    ps=Int[]; er=Float64[]; ts=Float64[]
    for p in PS
        p < 2 && continue
        ρ = spectralRadiusOfMapping_collocation(prob,1.0,p;S=S)
        e = abs(ρ-1)
        push!(ps,p); push!(er,max(e,1e-16)); push!(ts,cputime(S,p))
        @printf("GL%d p=%3d  |ρ-1|=%.3e  (%.2e s)\n",S,p,e,ts[end]); flush(stdout)
        count(<(2e-12), er) ≥ 3 && break     # a few points past the floor
    end
    data[S]=(p=ps,e=er,t=ts)
end

# ── figure: error vs resolution (left) and error vs CPU time (right) ──
p1 = plot(xscale=:log10,yscale=:log10,framestyle=:box,xlabel="p  (steps per delay)",
          ylabel="|ρ(H) − 1|   (error vs exact χ=0 boundary)",
          title="convergence onto the Iklodi–Dankowicz exact boundary",titlefontsize=10,
          legend=:bottomleft,guidefontsize=10,tickfontsize=8,legendfontsize=8)
for (ord,anch) in ((2,3e-1),(4,3e-2),(6,3e-3),(8,3e-4),(10,3e-5))
    pg=[2.0,128.0]; eg=anch.*(2.0./pg).^ord
    plot!(p1,pg,max.(eg,1e-16),color=:gray80,ls=:dot,lw=1,label="")
end
hline!(p1,[2e-11],color=:gray,ls=:dashdot,lw=1,label="solver floor")
p2 = plot(xscale=:log10,yscale=:log10,framestyle=:box,xlabel="CPU time [s]",
          ylabel="|ρ(H) − 1|",title="accuracy per unit CPU time",titlefontsize=10,
          legend=false,guidefontsize=10,tickfontsize=8)
for S in 1:5
    d=data[S]
    plot!(p1,d.p,d.e,marker=:utriangle,ms=4,lw=1.6,color=scol[S],label="GL$S (order $(2S))")
    plot!(p2,max.(d.t,1e-5),d.e,marker=:utriangle,ms=4,lw=1.6,color=scol[S],label="")
end
plt=plot(p1,p2,layout=(1,2),size=(1050,440),dpi=200,left_margin=5Plots.mm,bottom_margin=5Plots.mm)
savefig(plt, joinpath(@__DIR__,"cmp_iklodi_accuracy.png"))
savefig(plt, joinpath(@__DIR__,"cmp_iklodi_accuracy.pdf"))

println("\n══ measured slopes (order) ══")
for S in 1:5
    d=data[S]; idx=findall(>(3e-11),d.e)
    length(idx)<3 && continue
    i=idx[end-2:end]
    sl=log(d.e[i[1]]/d.e[i[3]])/log(d.p[i[3]]/d.p[i[1]])
    @printf("  GL%d: slope ≈ %.2f  (nominal %d)\n",S,sl,2S)
end
println("done — cmp_iklodi_accuracy.png")
