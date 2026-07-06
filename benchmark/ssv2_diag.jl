# Diagnose SSDM-vs-MC variance gap at (Ω0,w)=(1.0,0.30), 2-DOF SSV model.
# (1) classical M2 vs MF vs factored at coarse grid  (2) factored resolution sweep
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
BLAS.set_num_threads(4)

const N_TEETH=2; const PHI_EX=π/4; const Kr=0.3
const RVA=0.25; const NT=10; const ζ=0.02
const Ω0=1.0; const wdc=0.30
const Tssv = NT*2π/Ω0

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

function build(σcv, σav)
    Af(t) = SMatrix{4,4}([Z2 I2; (-I2 .- wdc.*HS(t)) (-2ζ).*I2])
    Bf(t) = SMatrix{4,4}([Z2 Z2; (wdc.*HS(t)) Z2])
    af(t) = SMatrix{4,4}([Z2 Z2; ((-σcv*wdc).*HS(t)) Z2])
    bf(t) = SMatrix{4,4}([Z2 Z2; ((σcv*wdc).*HS(t)) Z2])
    LDDEProblem(ProportionalMX(Af), [DelayMX(τf, Bf)],
        [stCoeffMX(1,ProportionalMX(af))], [stCoeffMX(1,DelayMX(τf, bf))],
        Additive(4), [stAdditive(2,Additive(@SVector [0.,0.,σav,0.]))])
end

function factored(lddep, nst)
    Δt = Tssv/nst
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τmax;
                                n_steps=nst, calculate_additive=true)
    ρ = spectralRadiusOfMapping_MF_factored(rst)
    v = fixPointOfMapping_MF_factored(rst)[1]
    (ρ, v)
end

# ── (1) three routes at coarse resolution ──
println("== route cross-check (coarse nst=100) ==")
lddep = build(0.20, 0.10)
nst_c = 100; Δt_c = Tssv/nst_c
rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt_c), τmax;
                            n_steps=nst_c, calculate_additive=true)
t=@elapsed begin
    dmM2 = DiscreteMapping_M2(rst)
    ρ_cl = spectralRadiusOfMapping(dmM2)
    v_cl = fixPointOfMapping(dmM2)[1]
end
@printf("classical M2 : ρ=%.8f  Var=%.8e  (%.1fs)\n", ρ_cl, v_cl, t)
ρ_fa, v_fa = factored(lddep, nst_c)
@printf("MF factored  : ρ=%.8f  Var=%.8e\n", ρ_fa, v_fa)
flush(stdout)

# ── (2) factored resolution sweep, full noise ──
println("== factored resolution sweep (full noise) ==")
for nst in (360, 720, 1440, 2880, 5760)
    t=@elapsed ((ρ,v) = factored(lddep, nst))
    @printf("nst=%5d (Δt=%.5f, r≈%d): ρ=%.6f  Var=%.6e  (%.0fs)\n",
            nst, Tssv/nst, Int(ceil(τmax/(Tssv/nst))), ρ, v, t)
    flush(stdout)
end

# ── (3) additive-only (σc=0) sweep ──
println("== factored sweep, σc=0 (additive only) ==")
lddep0 = build(0.0, 0.10)
for nst in (360, 1440, 5760)
    t=@elapsed ((ρ,v) = factored(lddep0, nst))
    @printf("nst=%5d: ρ=%.6f  Var=%.6e  (%.0fs)\n", nst, ρ, v, t)
    flush(stdout)
end
println("done")
