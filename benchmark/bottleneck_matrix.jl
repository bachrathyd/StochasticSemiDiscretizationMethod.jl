# =============================================================================
# Bottleneck matrix: (SD | GL2 | GL10) × (β≡0 | β large) × (d=2 | d=8) × (point | map)
#
# For every combination this measures, at ACCURACY-MATCHED resolution (smallest p
# with |ρ-ρ*|/|ρ*| ≤ 1e-4 — the practical chart accuracy):
#   point: cold wall time, allocation, GC share, engine memory footprint,
#          build/solve split (GL);
#   map:   12×12 parameter grid at that p, under four threading strategies
#          (serial | outer-threads×serial-build | nested | inner-build-only),
#          wall time + allocation + GC share.
# Methods: SD = ClassicalSD(2) (order 1), GL2 = Collocation(1) (order 2),
#          GL10 = Collocation(5) (order 10). Smooth T-periodic problems
#          (interrupted-cut coefficients are a separate, classical-only story).
# Incremental CSV: benchmark/bottleneck_matrix.csv.  Run with `julia -t auto`.
# =============================================================================
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod, StaticArrays, LinearAlgebra, Printf, DelimitedFiles
const SSM = StochasticSemiDiscretizationMethod
BLAS.set_num_threads(1)

const CSV = joinpath(@__DIR__, "bottleneck_matrix.csv")
const TARGET = 1e-4                      # practical relative-accuracy target in ρ

# ---------------- problems (type-stable: const params, @SMatrix coeffs) -------
# d=2 SSV turning; β knob scales the delayed multiplicative (regenerative) noise.
const RVA=0.1; const RVF=0.1; const ζ2=0.05; const σa2=0.1; const σc2=0.25
const T2=2π/RVF
function prob_d2(Ω0::Float64, w::Float64, βs::Float64)
    τf(t)=(2π)/(Ω0*(1.0+RVA*sin(RVF*t)))
    LDDEProblem(ProportionalMX(t->@SMatrix [0.0 1.0; -(1.0+w) -2ζ2]),
        [DelayMX(τf, t->@SMatrix [0.0 0.0; w 0.0])],
        [stCoeffMX(1,ProportionalMX(t->@SMatrix [0.0 0.0; -σc2*(1.0+w) 0.0]))],
        [stCoeffMX(1,DelayMX(τf, t->@SMatrix [0.0 0.0; βs*σc2*w 0.0]))],
        Additive(2),[stAdditive(1,Additive(@SVector [0.0,σa2]))],1)
end
# d=8: 4-mass oscillator chain, parametric stiffness modulation, delayed feedback
# w·q1(t-τ(t)) forcing mass 4; present multiplicative noise on the same force row;
# delayed multiplicative noise scaled by βs; additive force noise on mass 4.
const ζ8=0.05; const κ=1.0; const μmod=0.3; const σa8=0.1; const σc8=0.25
const T8=1.0
function prob_d8(km::Float64, w::Float64, βs::Float64)
    τf(t)=0.30+0.06*sin(2π*t/T8)
    A(t)=begin
        k1=κ*(1.0+μmod*cos(2π*t/T8))*km
        SMatrix{8,8,Float64}(
        # columns q1..q4, v1..v4 (column-major)
         0.0,0.0,0.0,0.0, -(k1+κ), κ, 0.0, 0.0,
         0.0,0.0,0.0,0.0,  κ, -2κ, κ, 0.0,
         0.0,0.0,0.0,0.0,  0.0, κ, -2κ, κ,
         0.0,0.0,0.0,0.0,  0.0, 0.0, κ, -κ,
         1.0,0.0,0.0,0.0, -2ζ8, 0.0, 0.0, 0.0,
         0.0,1.0,0.0,0.0,  0.0, -2ζ8, 0.0, 0.0,
         0.0,0.0,1.0,0.0,  0.0, 0.0, -2ζ8, 0.0,
         0.0,0.0,0.0,1.0,  0.0, 0.0, 0.0, -2ζ8)
    end
    B(t)=begin
        M=zeros(MMatrix{8,8,Float64}); M[8,1]=w; SMatrix(M)          # v̇4 += w·q1(t-τ)
    end
    α(t)=begin
        M=zeros(MMatrix{8,8,Float64}); M[8,1]=-σc8*w; SMatrix(M)
    end
    β(t)=begin
        M=zeros(MMatrix{8,8,Float64}); M[8,1]=βs*σc8*w; SMatrix(M)
    end
    LDDEProblem(ProportionalMX(A),[DelayMX(τf,B)],
        [stCoeffMX(1,ProportionalMX(α))],[stCoeffMX(1,DelayMX(τf,β))],
        Additive(8),[stAdditive(1,Additive(@SVector [0.,0.,0.,0.,0.,0.,0.,σa8]))],1)
end

mkprob(d,βs; θ1=0.87, θ2=0.4) = d==2 ? (prob_d2(θ1,θ2,βs), T2) : (prob_d8(θ1,θ2,βs), T8)
method_of(m) = m=="SD" ? ClassicalSD(2) : m=="GL2" ? Collocation(1) : Collocation(5)
ρof(prob,T,p,m; kw...) = spectralRadiusOfMoment(prob,T,p; method=method_of(m), verbosity=0, kw...)
# ladder-safe: an infeasible resolution (e.g. τ < Δt at the coarse end) is "no value"
ρtry(prob,T,p,m) = try; ρof(prob,T,p,m); catch; NaN; end

# ---------------- incremental CSV --------------------------------------------
have=Dict{String,Vector{Float64}}()
if isfile(CSV)
    for ln in readlines(CSV)[2:end]
        c=split(ln,','); have[String(c[1])]=parse.(Float64, c[2:end])
    end
end
function put(key, vals...)
    have[key]=collect(Float64, vals)
    open(CSV,"w") do io
        println(io,"key,v1,v2,v3,v4,v5,v6")
        for (k,v) in sort(collect(have))
            println(io, k*","*join(vcat(v, fill(NaN, 6-length(v))), ","))
        end
    end
end
tmin(f,n=3)=minimum(f() for _ in 1:n)

# ---------------- point study ------------------------------------------------
const LADDERS=Dict(
    ("SD",2)=>[100,200,400,800,1600,3200,6400,12800],
    ("GL2",2)=>[20,40,80,160,320,640,1280],
    ("GL10",2)=>[8,12,16,24,32,48,64,96],
    ("SD",8)=>[100,200,400,800,1600,3200],
    ("GL2",8)=>[20,40,80,160,320],
    ("GL10",8)=>[8,12,16,24,32,48])
richardson(v)=(q2=(v[2]-v[1])/(v[3]-v[2]); v[3]+(v[3]-v[2])/(q2-1))

for d in (2,8), βs in (0.0,1.0)
    tag(m)="point_$(m)_d$(d)_b$(Int(βs))"
    # reference: Richardson over the three finest GL10 rungs (cached)
    rkey="ref_d$(d)_b$(Int(βs))"
    if !haskey(have,rkey)
        prob,T=mkprob(d,βs)
        lad=LADDERS[("GL10",d)][end-2:end]
        vals=[ρof(prob,T,p,"GL10") for p in lad]
        put(rkey, richardson(vals))
        @printf("%s = %.10f  (GL10 p=%s)\n", rkey, have[rkey][1], lad); flush(stdout)
    end
    ρref=have[rkey][1]
    for m in ("SD","GL2","GL10")
        haskey(have,tag(m)) && continue
        prob,T=mkprob(d,βs)
        pstar=0; ρp=NaN
        for p in LADDERS[(m,d)]
            ρp=ρtry(prob,T,p,m)
            isnan(ρp) && continue
            abs(ρp-ρref)/abs(ρref) ≤ TARGET && (pstar=p; break)
        end
        if pstar==0
            put(tag(m), -1, NaN, NaN, NaN, NaN)     # target not reached on ladder
            @printf("  %-22s TARGET NOT REACHED (last err=%.1e)\n", tag(m), abs(ρp-ρref)/abs(ρref)); flush(stdout)
            continue
        end
        t=tmin(()->@elapsed(ρof(prob,T,pstar,m)), 3)
        al=@allocated ρof(prob,T,pstar,m)
        gct=(@timed ρof(prob,T,pstar,m)).gctime
        # engine memory + build/solve split (GL only)
        esz=NaN; tb=NaN; tsv=NaN
        if m!="SD"
            S = m=="GL2" ? 1 : 5
            pb=SSM._collocation_prob(prob,T)
            eng=SSM.build_vT(pb,S,pstar)
            esz=Base.summarysize(eng)/2^20
            tb=tmin(()->(@elapsed SSM.build_vT(pb,S,pstar)), 3)
            tsv=tmin(()->(@elapsed SSM.rho_H_krylov_v9m(eng)), 3)
        end
        put(tag(m), pstar, t, al/2^20, 100*gct/max(t,1e-12), esz, m=="SD" ? NaN : tb)
        @printf("  %-22s p*=%5d  t=%7.3fs  alloc=%8.1fMB  gc=%4.1f%%  engMB=%6.1f  build=%6.3fs solve=%6.3fs\n",
                tag(m), pstar, t, al/2^20, 100*gct/max(t,1e-12), esz, tb, tsv); flush(stdout)
    end
end

# ---------------- map study --------------------------------------------------
# 12×12 grid; θ1×θ2 around the nominal point. Strategies:
#   ser   serial, build_parallel=false
#   outer Threads.@threads over points, build_parallel=false
#   nest  Threads.@threads over points, build_parallel=true (nested)
#   inner serial over points, build_parallel=true
const G1=range(0.80,0.95;length=12)
const G2=range(0.25,0.50;length=12)
function runmap(m,d,βs,pstar,strategy)
    vals=zeros(length(G1),length(G2))
    mk(i,j)=mkprob(d,βs; θ1 = d==2 ? G1[i] : 0.5+0.5*G1[i], θ2=G2[j])
    if strategy in (:ser,:inner)
        bp = strategy==:inner
        for i in eachindex(G1), j in eachindex(G2)
            prob,T=mk(i,j); vals[i,j]=ρof(prob,T,pstar,m; build_parallel=bp)
        end
    else
        bp = strategy==:nest
        idx=[(i,j) for i in eachindex(G1), j in eachindex(G2)]
        Threads.@threads for k in eachindex(idx)
            (i,j)=idx[k]; prob,T=mk(i,j)
            vals[i,j]=ρof(prob,T,pstar,m; build_parallel=bp)
        end
    end
    sum(vals)
end
for d in (2,8), βs in (0.0,1.0), m in ("SD","GL2","GL10")
    pk="point_$(m)_d$(d)_b$(Int(βs))"
    haskey(have,pk) || continue
    pstar=Int(have[pk][1]); pstar>0 || continue
    tpt=have[pk][2]
    d==8 && m=="GL10" && tpt>2.0 && continue        # cap projected map cost
    key="map_$(m)_d$(d)_b$(Int(βs))"
    haskey(have,key) && continue
    # warm compile: one point per build_parallel branch (full-map warmups are waste)
    let (prob,T)=mkprob(d,βs)
        ρof(prob,T,pstar,m; build_parallel=false); ρof(prob,T,pstar,m; build_parallel=true)
    end
    res=Float64[]
    for strat in (:ser,:outer,:nest,:inner)
        st=@timed runmap(m,d,βs,pstar,strat)
        push!(res, st.time); push!(res, 100*st.gctime/st.time)
        @printf("  %-20s %-6s  %7.2fs  gc=%4.1f%%\n", key, strat, st.time, 100*st.gctime/st.time); flush(stdout)
    end
    # v1..v4 = wall(ser,outer,nest,inner); v5,v6 = GC%(ser,outer)
    put(key, res[1],res[3],res[5],res[7],res[2],res[4])
end
println("BOTTLENECK MATRIX DONE")
