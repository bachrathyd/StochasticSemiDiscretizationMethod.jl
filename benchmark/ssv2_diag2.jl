# Arbitration test: w→0 limit has exact answer Var(x)=σa²/(4ζ)=0.125.
# Compare factored & classical fixed points at d=4 and d=2 against it.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
BLAS.set_num_threads(4)

const N_TEETH=2; const PHI_EX=π/4; const Kr=0.3
const RVA=0.25; const NT=10; const ζ=0.02; const σa=0.10
const Ω0=1.0; const Tssv=NT*2π/Ω0
const wdc=1e-12                      # effectively no cutting

φ0fun(t) = Ω0*t - (Ω0*RVA*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)
function Hdir(t)
    φ0 = φ0fun(t); h11=0.0; h12=0.0; h21=0.0; h22=0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        φ ≤ PHI_EX || continue
        s,c = sincos(φ)
        a1=(c+Kr*s); a2=(s-Kr*c)
        h11+=a1*s; h12+=a1*c; h21+=-a2*s; h22+=-a2*c
    end
    (h11,h12,h21,h22)
end
τf(t) = (2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
const τmax = (2π/N_TEETH)/(Ω0*(1-RVA))
const Z2=@SMatrix zeros(2,2); const I2=SMatrix{2,2}(1.0I)
HS(t) = begin (a,b,c,d)=Hdir(t); @SMatrix [a b; c d] end

println("exact reference: Var(x) = σa²/(4ζ) = ", σa^2/(4ζ))

# ---- d=4 (2-DOF), w≈0 ----
Af4(t) = SMatrix{4,4}([Z2 I2; (-I2 .- wdc.*HS(t)) (-2ζ).*I2])
Bf4(t) = SMatrix{4,4}([Z2 Z2; (wdc.*HS(t)) Z2])
z4(t) = @SMatrix zeros(4,4)
p4 = LDDEProblem(ProportionalMX(Af4), [DelayMX(τf, Bf4)],
    [stCoeffMX(1,ProportionalMX(z4))], [stCoeffMX(1,DelayMX(τf, z4))],
    Additive(4), [stAdditive(2,Additive(@SVector [0.,0.,σa,0.]))])
nst=360; Δt=Tssv/nst
r4 = SSDM.calculateResults(p4, SemiDiscretization(2, Δt), τmax;
                           n_steps=nst, calculate_additive=true)
@printf("d=4 factored : Var=%.6e\n", fixPointOfMapping_MF_factored(r4)[1]); flush(stdout)
nstc=100
r4c = SSDM.calculateResults(p4, SemiDiscretization(2, Tssv/nstc), τmax;
                            n_steps=nstc, calculate_additive=true)
@printf("d=4 classical: Var=%.6e\n", fixPointOfMapping(DiscreteMapping_M2(r4c))[1]); flush(stdout)

# ---- d=2 (1-DOF), w≈0 ----
h11f(t) = Hdir(t)[1]
Af2(t) = @SMatrix [0. 1.; (-1-wdc*h11f(t)) -2ζ]
Bf2(t) = @SMatrix [0. 0.; (wdc*h11f(t)) 0.]
z2(t) = @SMatrix zeros(2,2)
p2 = LDDEProblem(ProportionalMX(Af2), [DelayMX(τf, Bf2)],
    [stCoeffMX(1,ProportionalMX(z2))], [stCoeffMX(1,DelayMX(τf, z2))],
    Additive(2), [stAdditive(2,Additive(@SVector [0.,σa]))])
r2 = SSDM.calculateResults(p2, SemiDiscretization(2, Δt), τmax;
                           n_steps=nst, calculate_additive=true)
@printf("d=2 factored : Var=%.6e\n", fixPointOfMapping_MF_factored(r2)[1]); flush(stdout)
r2c = SSDM.calculateResults(p2, SemiDiscretization(2, Tssv/nstc), τmax;
                            n_steps=nstc, calculate_additive=true)
@printf("d=2 classical: Var=%.6e\n", fixPointOfMapping(DiscreteMapping_M2(r2c))[1]); flush(stdout)

# ---- d=4, additive in BOTH velocity components ----
p4b = LDDEProblem(ProportionalMX(Af4), [DelayMX(τf, Bf4)],
    [stCoeffMX(1,ProportionalMX(z4))], [stCoeffMX(1,DelayMX(τf, z4))],
    Additive(4), [stAdditive(2,Additive(@SVector [0.,0.,σa,σa]))])
r4b = SSDM.calculateResults(p4b, SemiDiscretization(2, Δt), τmax;
                            n_steps=nst, calculate_additive=true)
m4b = fixPointOfMapping_MF_factored(r4b)
@printf("d=4 factored, σa in both vx,vy: Var(x)=%.6e\n", m4b[1])
println("done")
