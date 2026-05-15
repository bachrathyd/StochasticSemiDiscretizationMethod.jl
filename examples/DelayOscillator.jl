using StochasticSemiDiscretizationMethod
using StaticArrays
using MDBM
using Plots
using LaTeXStrings

gr();

function createSLDOProblem(A,B,ζ,α,β,σ)
    AMx =  ProportionalMX(@SMatrix [0. 1.;-A -2ζ]);
    τ1=2π 
    BMx1 = DelayMX(τ1,@SMatrix [0. 0.; B 0.]);
    cVec = Additive(2)
    noiseID = 1
    αMx1 = stCoeffMX(noiseID,ProportionalMX(@SMatrix [0. 0.; α 0.]))
    βMx11 = stCoeffMX(noiseID,DelayMX(τ1,@SMatrix [0. 0.; β 0.]))
    σVec = stAdditive(1,Additive(@SVector [0., σ]))
    LDDEProblem(AMx,[BMx1],[αMx1],[βMx11],cVec,[σVec])
end

method=SemiDiscretization(5, 2π/30)
τmax=2π + 100eps()

# Point calculation
sldo_lddep=createSLDOProblem(1.,0.1,0.1,0.1,0.1,0.5);
mapping=DiscreteMapping_M2_MF(sldo_lddep,method,τmax,n_steps=30,calculate_additive=true);

@show spectralRadiusOfMapping_MF(mapping);
r = div(mapping.rst.n, 2) - 1
D1 = (r+1)*2
statM2=VecToCovMx(fixPointOfMapping_MF(mapping), D1);
@show statM2[1,1] |> sqrt;

# Map calculation
function foo(A::Float64, B::Float64)::Float64
    lddep = createSLDOProblem(A,B,0.05,0.3*A,0.3*B,0.)
    rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τmax, n_steps=30)
    dm = DiscreteMapping_M2_MF(rst)
    return log(spectralRadiusOfMapping_MF(dm))
end

axis=[Axis(-1.0:0.25:5.0,:A),
    Axis(LinRange(-1.5,1.5,12),:B)]

iteration=4;
mdbm_prob = MDBM_Problem(foo, axis)
solve!(mdbm_prob, iteration, verbosity=2)
stab_border_points=getinterpolatedsolution(mdbm_prob);

p = scatter(stab_border_points...,xlim=(-1.,5.),ylim=(-1.5,1.4),
    label="",title="2nd moment stability border of the Delay Oscillator",xlabel=L"A",ylabel=L"$B$",
    guidefontsize=14,tickfont = font(10),markersize=2,markerstrokewidth=0)
display(p)
savefig(p, "assets/DelayOscillatorStability.png")
