5 + 5

using StochasticSemiDiscretizationMethod
using Plots
gr();

function createSLDOProblem(k, ζ, P, D, τ, σ)
    AMx = ProportionalMX(@SMatrix [0.0 1.0; -k -2ζ])
    #  AMx =  ProportionalMX(@SMatrix [0. 1.;-k. 0.0]);
    #τ1=2π 
    BMx1 = DelayMX(τ, @SMatrix [0.0 0.0; -P -D])
    cVec = Additive(2)
    noiseID = 1
    αMx1 = stCoeffMX(noiseID, ProportionalMX(@SMatrix [0.0 0.0; 0.0 0.0]))
    βMx11 = stCoeffMX(noiseID, DelayMX(τ, @SMatrix [0.0 0.0; 0.0 0.0]))
    # αMx1 = stCoeffMX(noiseID,ProportionalMX(@SMatrix [0. 0.; α 0.]))
    # βMx11 = stCoeffMX(noiseID,DelayMX(τ1,@SMatrix [0. 0.; β 0.]))
    σVec = stAdditive(1, Additive(@SVector [0.0, σ]))
    LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end
k = -10.0
ζ = 0.0
P = 15.0
D = 5.0
τ = 0.2
σ = 0.1

function foo(k, ζ, P, D, τ, σ)
    SLDOP_lddep = createSLDOProblem(k, ζ, P, D, τ, σ) # LDDE problem for Hayes equation
    method = SemiDiscretization(5, (τ + 100eps()) / 15) # 5th order semi discretization with Δt=2π/10
    τmax = τ# the largest τ of the system
    # Second Moment mapping
    mapping = DiscreteMapping_M2(SLDOP_lddep, method, τmax, n_steps=10, calculate_additive=true) #The discrete mapping of the system

    MUMAX = spectralRadiusOfMapping(mapping) # spectral radius ρ of the mapping matrix (ρ>1 unstable, ρ<1 stable)
    statM2 = VecToCovMx(fixPointOfMapping(mapping), length(mapping.M1_Vs[1])) # stationary second moment matrix of the hayes equation (equilibrium position)
    return (MUMAX, P * statM2[2, end-1] + D * statM2[2, end], statM2)

end
@time foo(1.0, 0.01, 0.1, 0.1, 0.1, 0.1)
@time foo(1.0, 0.01, 0.1, 0.1, 0.1, 0.1)



#AA= foo(1.0,0.01,0.1,0.1,0.1,0.1)[2]
#surface(AA[:,:])
#surface(AA[1:2:end,1:2:end])
#surface(AA[1:2:end,2:2:end])
#surface(AA[2:2:end,1:2:end])
#surface(-AA[2:2:end,2:2:end])

Pv = LinRange(9.0, 20.0,100);
Dv = LinRange(-0.0, 10.0, 80);
Pv = LinRange(9.0, 20.0,25);
Dv = LinRange(-0.0, 10.0, 20);
@time Sols = [foo(k, ζ, Pi, Di, τ, σ) for Pi in Pv, Di in Dv];

mus = [Sol[1] * (Sol[1] < 1) for Sol in Sols];
Pow = [Sol[2] * (Sol[1] < 1) for Sol in Sols];
Pow = [Sol[2] * (Sol[1] < 1) * (Sol[2] > 0) for Sol in Sols];
#Pow = [Sol[2] * (Sol[1] < 1) for Sol in Sols];
#Pow=[Sol[2]*(Sol[1]<1)  for Sol in Sols];
Asteady = [-maximum([-0.005, -Sol[3][1, 1] * (Sol[1] < 0.999)]) for Sol in Sols];
#Asteady=[-maximum([-0.001,-Sol[3][2,2]*(Sol[1]<1)]) for Sol in Sols];

contourf(Pv, Dv, mus', levels=20)
heatmap(Pv, Dv, mus')
#@show contourf(Pv,Dv,mus',levels=[0.995,1.0,1.005])
#-----------
maximum(Asteady)
maximum(-Asteady)
# contourf(Pv,Dv,Asteady')
# contourf(Pv,Dv,log.(abs.(Asteady')))
contourf(Pv, Dv, log.((Asteady')))

#@show surface(Pv,Dv,log.((Asteady')))

##--------------------------------------------------------------
#@show maximum(Pow[:])
#@show -maximum(-Pow[:])

contourf(Pv, Dv, Pow', levels=30)


heatmap(Pv, Dv, Pow')
5 + 5

begin
    plot()
    for i in 1:2:80#size(Dv)[1]
        plot!(Pv, Pow[:, i])
    end
    plot!(ylim=[-0.040, 0.010])
end

5 + 5
#surface(Pv,Dv,Pow',ylim=[-20,20])
##--------------------------------------------------------------

## Stability chart for the Delay Oscillator
#using MDBM
#using Plots
#gr();
#using LaTeXStrings
#
#method=SemiDiscretization(5,2π/30);
#τmax=2π+100eps()
#idxs=[1,2,3:2:StochasticSemiDiscretizationMethod.rOfDelay(τmax,method)*2...]
#
## ζ=0.05, α=0.3*A, β=0.3*B
#foo(A,B) = log(spectralRadiusOfMapping(DiscreteMapping_M2(createSLDOProblem(A,B,0.05,0.3*A,0.3*B,0.),method,τmax,idxs,
#    n_steps=30),nev=8)); # No additive term calculated
#
#axis=[Axis(-1.0:0.6:5.0,:A),
#    Axis(LinRange(-1.5,1.5,12),:B)]
#
#iteration=4;
#stab_border_points=getinterpolatedsolution(solve!(MDBM_Problem(foo,axis),iteration));
#
#scatter(stab_border_points...,xlim=(-1.,5.),ylim=(-1.5,1.4),
#    label="",title="2nd moment stability border of the Delay Oscillator",xlabel=L"A",ylabel=L"$B$",
#    guidefontsize=14,tickfont = font(10),markersize=2,markerstrokewidth=0)
#