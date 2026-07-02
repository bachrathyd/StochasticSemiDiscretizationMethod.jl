# =============================================================================
# Factored-operator vs dense (previous) MF second-moment: timing across p and d.
#
# Same fully-periodic delayed-stochastic chain-of-oscillators (all matrices
# periodic + delayed + STRONG multiplicative noise) at increasing DoF.
# For each (d, p) we time the coefficient BUILD and the ρ(H) SOLVE for both
# methods (warmed, best-of-2), verify ρ agrees, and record. The dense path is
# only attempted where its d²×d² StaticArrays coefficients still compile;
# the factored path continues to d = 100+ (10-DoF … 50-DoF … 100-DoF).
#
# Run:  julia -t 36 --project=. benchmark/benchmark_factored_vs_dense.jl
# Out:  benchmark/factored_vs_dense.csv, .png  (saved after every point)
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Plots, Printf
BLAS.set_num_threads(1)

# strong noise (αs) — makes the covariance non-trivial and the comparison honest
function chain(n; k0=3.0, εk=0.8, ζ=0.05, b0=0.4, αs=0.25)
    d = 2n
    Kmat(t)=begin k=k0*(1+εk*cos(0.5t)); K=zeros(n,n)
        for i in 1:n; K[i,i]=2k; i>1&&(K[i,i-1]=-k); i<n&&(K[i,i+1]=-k); end; K end
    Af(t)=begin A=zeros(d,d); A[1:n,n+1:d]=Matrix(I,n,n); A[n+1:d,1:n]=-Kmat(t); A[n+1:d,n+1:d]=-2ζ*Matrix(I,n,n); A end
    Bf(t)=begin B=zeros(d,d); b=b0*(1+0.4cos(0.5t)); for i in 1:n; B[n+i,i]=b; end; B end
    af(t)=begin A=zeros(d,d); A[n+1:d,1:n]=-αs*Kmat(t); A end
    bf(t)=begin B=zeros(d,d); b=αs*b0*(1+0.4cos(0.5t)); for i in 1:n; B[n+i,i]=b; end; B end
    LDDEProblem(ProportionalMX(Af), [DelayMX(2π,Bf)], [stCoeffMX(1,ProportionalMX(af))],
        [stCoeffMX(1,DelayMX(2π,bf))], Additive(d), [stAdditive(1,Additive(zeros(d)))])
end

timeit(f) = (t0=time(); v=f(); (time()-t0, v))
best2(f) = (t,v=timeit(f); t2,_=timeit(f); (min(t,t2),v))

# dense path builds coefficients via SMatrix — attempt only up to this d
const DENSE_DMAX = 10
const rows = NamedTuple[]
csv = joinpath(@__DIR__, "factored_vs_dense.csv")
png = joinpath(@__DIR__, "factored_vs_dense.png")

function save(rows)
    open(csv,"w") do io
        println(io,"ndof,d,p,D,build_dense,solve_dense,build_fact,solve_fact,rho,rel")
        for r in rows
            @printf(io,"%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.10f,%.2e\n",
                r.ndof,r.d,r.p,r.D,r.bd,r.sd,r.bf,r.sf,r.rho,r.rel)
        end
    end
    # solve-time vs p, dense vs factored, per d
    plt = plot(title="ρ(H) solve time: dense vs factored (chain, strong noise)",
               xlabel="p", ylabel="solve time [s]", xscale=:log10, yscale=:log10, legend=:topleft)
    cols = Dict(2=>:blue,4=>:red,6=>:green,8=>:purple,10=>:orange,20=>:brown,40=>:black,100=>:magenta)
    for dd in sort(unique(r.d for r in rows))
        sel = filter(r->r.d==dd, rows); isempty(sel) && continue
        pv=[r.p for r in sel]
        ds=[r.sd for r in sel if !isnan(r.sd)]; pvd=[r.p for r in sel if !isnan(r.sd)]
        !isempty(pvd) && plot!(plt, pvd, ds, marker=:circle, color=get(cols,dd,:gray), label="dense d=$dd")
        plot!(plt, pv, [r.sf for r in sel], marker=:star5, ls=:dash, color=get(cols,dd,:gray), label="factored d=$dd")
    end
    savefig(plt, png)
end

# grid: head-to-head where dense works, factored-only beyond
plan = [(2,[32,64,128,256]), (4,[32,64,128,256]), (6,[32,64,128]),
        (8,[32,64,128]), (10,[32,64]), (20,[32,64]), (50,[32]), (100,[24])]  # ndof
for (ndof, ps) in plan
    d = 2ndof
    println("═══ $ndof-DoF (d=$d) ═══"); flush(stdout)
    lddep = chain(ndof)
    for p in ps
        rst = SSDM.calculateResults(lddep, SemiDiscretization(2, 4π/p), 2π, n_steps=p)
        r = div(rst.n, d) - 1
        D = SSDM.CovVecIdx((r+1)*d).sectionStarts[end]
        # factored
        bf, dm_ignore = 0.0, nothing
        sf, ρf = best2(() -> spectralRadiusOfMapping_MF_factored(rst))
        # dense (only if feasible)
        bd = NaN; sd = NaN; ρd = NaN; rel = NaN
        if d <= DENSE_DMAX
            bd, dm = best2(() -> DiscreteMapping_M2_MF(rst))
            sd, ρd = best2(() -> spectralRadiusOfMapping_MF(dm))
            rel = abs(ρd-ρf)/abs(ρd)
        end
        push!(rows, (ndof=ndof, d=d, p=p, D=D, bd=bd, sd=sd, bf=NaN, sf=sf, rho=ρf, rel=rel))
        @printf("  p=%4d D=%9d | dense build %7.3f solve %7.3f | factored solve %7.3f | agree %.1e\n",
                p, D, bd, sd, sf, rel)
        flush(stdout); save(rows)
    end
end
println("done — $csv, $png")
