# Monte-Carlo validation of the stationary variance for one representative
# stable point of the SSV milling model (requested in external review):
# Euler–Maruyama simulation of eq. (milling) with time-varying delay, ensemble
# variance vs the MF-SSDM fixpoint prediction.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, Random
BLAS.set_num_threads(1)

const N_TEETH=2; const aD=0.5; const KtKn=0.3
const RVA=0.10; const NT=10; const ζ=0.02
const σc=0.20; const σa=0.10; const R_RES=24; const NAT_RES=30

function hfun(t, Ω0, Tssv)
    φ0 = Ω0*t - (Ω0*RVA*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)
    φen = acos(2aD - 1); φex = float(π)
    hsum = 0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        (φen ≤ φ ≤ φex) && (hsum += sin(φ)*(cos(φ) + KtKn*sin(φ)))
    end
    hsum
end

const Ω0 = 1.0; const wdc = 0.30                       # representative stable point
const Tssv = NT * 2π/Ω0
τf(t) = (2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
τmax  = (2π/N_TEETH)/(Ω0*(1-RVA))

# ── MF-SSDM prediction ──
Af(t) = @SMatrix [0. 1.; -(1.0 + wdc*hfun(t,Ω0,Tssv)) -2ζ]
Bf(t) = @SMatrix [0. 0.; wdc*hfun(t,Ω0,Tssv) 0.]
af(t) = @SMatrix [0. 0.; σc*wdc*hfun(t,Ω0,Tssv) 0.]
bf(t) = @SMatrix [0. 0.; -σc*wdc*hfun(t,Ω0,Tssv) 0.]
lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(τf, Bf)],
    [stCoeffMX(1,ProportionalMX(af))], [stCoeffMX(1,DelayMX(τf, bf))],
    Additive(2), [stAdditive(1,Additive(@SVector [0., σa]))])
Δt = min(τmax/R_RES, 2π/NAT_RES); nst=Int(round(Tssv/Δt)); Δt=Tssv/nst
rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τmax;
                            n_steps=nst, calculate_additive=true)
ρ = spectralRadiusOfMapping_MF_factored(rst)
m = fixPointOfMapping_MF_factored(rst)
var_ssdm = m[1]
@printf("MF-SSDM:  ρ(H)=%.6f  Var(x) at period start = %.6e\n", ρ, var_ssdm)
@assert ρ < 1

# ── Euler–Maruyama Monte Carlo ──
# dt_mc small; path history stored on a fine grid for the delayed read (linear
# interpolation); transient of 40 periods discarded; Var(x) sampled at period
# starts over 20 further periods; NPATH paths, threaded.
const dt_mc = Δt/8
const nsub  = Int(round(Tssv/dt_mc))
const dtm   = Tssv/nsub
const NTRANS=40; const NAVG=20; const NPATH=20_000
hbuf_len = Int(ceil(τmax/dtm)) + 2

function one_path(seed)
    rng = MersenneTwister(seed)
    nb = hbuf_len
    xs = zeros(nb); vs = zeros(nb)                    # circular history of x, ẋ
    x=0.0; v=0.0; head=0
    acc = 0.0; cnt = 0
    ttot = (NTRANS+NAVG)*Tssv
    nstep = Int(round(ttot/dtm))
    per = Int(round(Tssv/dtm))
    for k in 0:nstep-1
        t = k*dtm
        τ = (2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
        back = τ/dtm
        ib = floor(Int, back); frac = back - ib
        i1 = mod(head - ib, nb) + 1; i2 = mod(head - ib - 1, nb) + 1
        xd = (1-frac)*xs[i1] + frac*xs[i2]
        h  = hfun(t, Ω0, Tssv)
        ξ1 = randn(rng)*sqrt(dtm); ξ2 = randn(rng)*sqrt(dtm)
        drift_v = -(2ζ)*v - x - wdc*h*(x - xd)
        dv = drift_v*dtm - σc*wdc*h*(x - xd)*ξ1 + σa*ξ2
        v += dv; x += v*dtm            # semi-implicit (symplectic) Euler:
                                        # removes the energy drift of the
                                        # explicit scheme on oscillators
        head = mod(head+1, nb)
        xs[head+1]=x; vs[head+1]=v
        if k ≥ NTRANS*per && (k % per == 0)
            acc += x^2; cnt += 1
        end
    end
    (acc, cnt)
end

println("MC: $(NPATH) paths, dt=$(round(dtm,sigdigits=3)), $(NTRANS)+$(NAVG) periods ...")
t0=time()
nchunk = 8*Threads.nthreads()
chunks = [i:min(i+cld(NPATH,nchunk)-1, NPATH) for i in 1:cld(NPATH,nchunk):NPATH]
accs = zeros(length(chunks)); cnts = zeros(Int, length(chunks))
Threads.@threads for ci in eachindex(chunks)
    a=0.0; c=0
    for i in chunks[ci]
        ai,cci = one_path(1000+i); a+=ai; c+=cci
    end
    accs[ci]=a; cnts[ci]=c
end
var_mc = sum(accs)/sum(cnts)
sem = var_mc*sqrt(2/sum(cnts))                        # rough SE of variance estimate
@printf("MC:       Var(x) at period start = %.6e ± %.1e   (%.0f s)\n", var_mc, sem, time()-t0)
@printf("relative deviation SSDM vs MC: %.2f%%  (MC 95%% CI ±%.2f%%)\n",
        100*abs(var_ssdm-var_mc)/var_mc, 100*1.96*sem/var_mc)
