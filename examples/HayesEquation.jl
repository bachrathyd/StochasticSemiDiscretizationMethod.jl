using StochasticSemiDiscretizationMethod
using StaticArrays
using MDBM
using Plots
using LaTeXStrings
using LinearAlgebra; BLAS.set_num_threads(1)   # small/thin solves: single-thread BLAS is faster

gr();

function createHayesProblem(a,β)
    AMx =  ProportionalMX(a*ones(1,1));
    τ1=1. 
    BMx1 = DelayMX(τ1,zeros(1,1));
    cVec = Additive(1)
    noiseID = 1
    αMx1 = stCoeffMX(noiseID,ProportionalMX(zeros(1,1)))
    βMx11 = stCoeffMX(noiseID,DelayMX(τ1,β*ones(1,1)))
    σ = stAdditive(1,Additive(ones(1)))
    LDDEProblem(AMx,[BMx1],[αMx1],[βMx11],cVec,[σ])
end

method=SemiDiscretization(0,0.1);
τmax=1.

# Point calculation
hayes_lddep=createHayesProblem(-6.,2.);
mapping=DiscreteMapping_M2_MF(hayes_lddep,method,τmax,n_steps=10,calculate_additive=true);

@show spectralRadiusOfMapping_MF(mapping);
r = div(mapping.rst.n, 1) - 1
D1 = (r+1)*1
statM2=VecToCovMx(fixPointOfMapping_MF(mapping), D1);
@show statM2[1,1]

# Map calculation
function foo(a::Float64, b::Float64)::Float64
    lddep = createHayesProblem(a, b)
    rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τmax, n_steps=10)
    dm = DiscreteMapping_M2_MF(rst)
    return log(spectralRadiusOfMapping_MF(dm))
end

axis=[Axis(-9.0:1.0,:a),
    Axis(-5.0:5.0,:β)]

iteration=4;
mdbm_prob = MDBM_Problem(foo, axis)
solve!(mdbm_prob, iteration, verbosity=2)
stab_border_points=getinterpolatedsolution(mdbm_prob);

p = scatter(stab_border_points...,xlim=(-9.,1.),ylim=(-5.,5.),
    label="",title="2nd moment stability border of the Hayes equation",xlabel=L"A",ylabel=L"$\beta$",
    guidefontsize=14,tickfont = font(10),markersize=2,markerstrokewidth=0)
savefig(p, "assets/HayesStability.png")
