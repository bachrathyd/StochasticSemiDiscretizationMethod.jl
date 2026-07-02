# =============================================================================
# SELF-CONVERGENCE figure — stoch delay Mathieu 2nd-moment ρ(H), v6 engine (Krylov ρ).
# NO common reference: each GL(S) is measured against ITS OWN finest p:
#     selferr_S(p) = |ρ_S(p) − ρ_S(p_finest)|
# This isolates each method's INTERNAL convergence ORDER (slope), free of any common-reference
# offset (the common-ref figure showed a ~3e-4 v6-vs-SDM offset that is NOT a convergence stall).
# SDM q=2,4 included for comparison (their own finest). Output: mathieu_v6_selfconv.png + .csv
# =============================================================================
include(joinpath(@__DIR__,"cov_colloc_v6.jl"))
using Printf
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using Plots; gr()
using StochasticSemiDiscretizationMethod; const SSDM=StochasticSemiDiscretizationMethod
using StaticArrays

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1
Amat(t)=[0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA]; Bmat(t)=[0.0 0.0; Bval 0.0]
αmat(t)=[0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA]; βmat(t)=[0.0 0.0; ALPHA*Bval 0.0]
pb=Prob(2,PER,TAU, Amat,Bmat,αmat,βmat)
v6rho(S,p)=rho_H_krylov(build_v6(pb,S,p))
function sdm_rho(q,p)
    AMx=ProportionalMX(t->@SMatrix [0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA])
    BMx=DelayMX(TAU,@SMatrix [0.0 0.0; Bval 0.0])
    αMx=stCoeffMX(1,ProportionalMX(t->@SMatrix [0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA]))
    βMx=stCoeffMX(1,DelayMX(TAU,t->@SMatrix [0.0 0.0; ALPHA*Bval 0.0]))
    cV=Additive(@SVector [0.0,0.0]); σV=stAdditive(1,Additive(@SVector [0.0,0.0]))
    lddep=LDDEProblem(AMx,[BMx],[αMx],[βMx],cV,[σV])
    rst=SSDM.calculateResults(lddep,SemiDiscretization(q,PER/p),TAU;n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst);tol=1e-12)
end

# per-method p grids (v6 finest = high p via Krylov; SDM finest = 512)
grids = Dict(
  "v6GL1"=>(p->v6rho(1,p), round.(Int, exp10.(range(log10(6),log10(300),length=12)))),
  "v6GL2"=>(p->v6rho(2,p), round.(Int, exp10.(range(log10(6),log10(200),length=12)))),
  "v6GL3"=>(p->v6rho(3,p), round.(Int, exp10.(range(log10(6),log10(140),length=11)))),
  "v6GL4"=>(p->v6rho(4,p), round.(Int, exp10.(range(log10(6),log10(90), length=10)))),
  "v6GL5"=>(p->v6rho(5,p), round.(Int, exp10.(range(log10(6),log10(60), length=9)))),
  "SDM2" =>(p->sdm_rho(2,p), [8,16,32,64,128,256,512]),
  "SDM4" =>(p->sdm_rho(4,p), [8,16,32,64,128,256,512]),
)
ordr=["v6GL1","v6GL2","v6GL3","v6GL4","v6GL5","SDM2","SDM4"]
for m in ordr; (f,g)=grids[m]; grids[m]=(f,sort(unique(g))); end

results=Dict{String,Vector{Tuple{Int,Float64}}}()
for m in ordr
    f,ps=grids[m]; vals=Tuple{Int,Float64}[]
    for p in ps
        try v=f(p); push!(vals,(p,v)); @printf("%-6s p=%3d ρ=%.10f\n",m,p,v); flush(stdout)
        catch e; @warn "$m p=$p failed" e; flush(stdout); end
    end
    results[m]=vals
end

open(joinpath(@__DIR__,"mathieu_v6_selfconv.csv"),"w") do io
    println(io,"# stoch delay Mathieu 2nd moment v6 SELF-convergence (each method vs its own finest p)")
    println(io,"method,p,rho,self_err")
    for m in ordr
        vals=results[m]; isempty(vals)&&continue; ρf=vals[end][2]
        for (p,ρ) in vals; @printf(io,"%s,%d,%.12g,%.12g\n",m,p,ρ,abs(ρ-ρf)); end
    end
end

col=Dict("v6GL1"=>:dodgerblue,"v6GL2"=>:seagreen,"v6GL3"=>:purple,"v6GL4"=>:darkorange,"v6GL5"=>:red,
         "SDM2"=>:gray50,"SDM4"=>:black)
plt=plot(xlabel="p (lépés/periódus)",ylabel="|ρ(p) − ρ(saját legfinomabb)|  (önkonvergencia)",
    title="Stoch. delay Mathieu 2nd moment — ÖNKONVERGENCIA (közös referencia NÉLKÜL)\nminden eljárás a saját határértékéhez (jelen+múlt mult. zaj); v6 magasrendű vs SDM",
    xscale=:log10,yscale=:log10,legend=:bottomleft,lw=2,size=(980,720),titlefontsize=9)
for m in ordr
    vals=results[m]; length(vals)<2 && continue; ρf=vals[end][2]
    ps=[p for (p,_) in vals[1:end-1]]; es=[max(abs(ρ-ρf),1e-16) for (_,ρ) in vals[1:end-1]]
    lbl = startswith(m,"SDM") ? "SDM q=$(m[4])" : "GL$(m[5]) (rend $(2*parse(Int,m[5])))"
    ls = startswith(m,"SDM") ? :dash : :solid; mk= startswith(m,"SDM") ? :diamond : :circle
    plot!(plt,ps,es,label=lbl,marker=mk,linestyle=ls,color=col[m],ms=4,markerstrokewidth=0)
end
savefig(plt,joinpath(@__DIR__,"mathieu_v6_selfconv.png"))
println("saved mathieu_v6_selfconv.png + .csv"); flush(stdout)
println("V6SELFFIG DONE"); flush(stdout)
