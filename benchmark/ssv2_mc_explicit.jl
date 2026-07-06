# Monte-Carlo validation of the stationary variance for one representative
# stable point of the TWO-DOF SSV milling model (cross-coupled directional
# matrix; cf. ssv2dof_chart.jl). Semi-implicit (symplectic) Euler–Maruyama —
# the explicit variant is energy-drift-biased on oscillators (see
# ssv_mc_diagnostic.jl).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, Random
BLAS.set_num_threads(1)

const N_TEETH=2; const PHI_EX=π/4; const Kr=0.3
const RVA=0.25; const NT=10; const ζ=0.02
const σc=0.20; const σa=0.10; const R_RES=24; const NAT_RES=30
const Ω0 = 1.0; const wdc = 0.30
const Tssv = NT * 2π/Ω0

φ0fun(t) = Ω0*t - (Ω0*RVA*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)
function Hdir(t)
    φ0 = φ0fun(t)
    h11=0.0; h12=0.0; h21=0.0; h22=0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        φ ≤ PHI_EX || continue
        s,c = sincos(φ)
        a1 = (c + Kr*s); a2 = (s - Kr*c)
        h11 +=  a1*s; h12 +=  a1*c
        h21 += -a2*s; h22 += -a2*c
    end
    (h11,h12,h21,h22)
end

# ── MF-SSDM prediction (identical build to ssv2dof_chart.jl) ──
τf(t) = (2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
τmax  = (2π/N_TEETH)/(Ω0*(1-RVA))
# type-stable: 4×4 SMatrix column-major from scalars (no runtime block hvcat)
Af(t) = begin (h11,h12,h21,h22)=Hdir(t); SMatrix{4,4,Float64}(
    0.0, 0.0, -1-wdc*h11, -wdc*h21,
    0.0, 0.0, -wdc*h12, -1-wdc*h22,
    1.0, 0.0, -2ζ, 0.0,
    0.0, 1.0, 0.0, -2ζ) end
lower4(h11,h12,h21,h22, s) = SMatrix{4,4,Float64}(   # s·H in the lower-left block
    0.0, 0.0, s*h11, s*h21,
    0.0, 0.0, s*h12, s*h22,
    0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0)
Bf(t) = lower4(Hdir(t)..., wdc)
af(t) = lower4(Hdir(t)..., -σc*wdc)
bf(t) = lower4(Hdir(t)..., σc*wdc)
lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(τf, Bf)],
    [stCoeffMX(1,ProportionalMX(af))], [stCoeffMX(1,DelayMX(τf, bf))],
    Additive(4), [stAdditive(2,Additive(@SVector [0.,0.,σa,0.]))])
Δt = min(τmax/R_RES, 2π/NAT_RES); nst=Int(round(Tssv/Δt)); Δt=Tssv/nst
rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τmax;
                            n_steps=nst, calculate_additive=true)
ρ = spectralRadiusOfMapping_MF_factored(rst)
m = fixPointOfMapping_MF_factored(rst)
var_ssdm = m[1]
@printf("MF-SSDM (2-DOF):  ρ(H)=%.6f  Var(x) at period start = %.6e\n", ρ, var_ssdm)
@assert ρ < 1

# ── symplectic Euler–Maruyama Monte Carlo ──
const dt_mc = Δt/8
const nsub  = Int(round(Tssv/dt_mc))
const dtm   = Tssv/nsub
const NTRANS=40; const NAVG=20; const NPATH=4_000   # explicit-Euler bias probe
hbuf_len = Int(ceil(τmax/dtm)) + 2

function one_path(seed)
    rng = MersenneTwister(seed)
    nb = hbuf_len
    xs = zeros(nb); ys = zeros(nb)
    x=0.0; y=0.0; vx=0.0; vy=0.0; head=0
    acc = 0.0; cnt = 0
    nstep = Int(round((NTRANS+NAVG)*Tssv/dtm))
    per = Int(round(Tssv/dtm))
    for k in 0:nstep-1
        t = k*dtm
        τ = (2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
        back = τ/dtm
        ib = floor(Int, back); fr = back - ib
        i1 = mod(head - ib, nb) + 1; i2 = mod(head - ib - 1, nb) + 1
        xd = (1-fr)*xs[i1] + fr*xs[i2]
        yd = (1-fr)*ys[i1] + fr*ys[i2]
        (h11,h12,h21,h22) = Hdir(t)
        Δx = x - xd; Δy = y - yd
        Fx = h11*Δx + h12*Δy
        Fy = h21*Δx + h22*Δy
        ξ1 = randn(rng)*sqrt(dtm); ξ2 = randn(rng)*sqrt(dtm)
        dvx = (-2ζ*vx - x - wdc*Fx)*dtm - σc*wdc*Fx*ξ1 + σa*ξ2
        dvy = (-2ζ*vy - y - wdc*Fy)*dtm - σc*wdc*Fy*ξ1
        x  += vx*dtm; y += vy*dtm            # EXPLICIT Euler: positions with OLD v
        vx += dvx; vy += dvy                 # (energy-drift bias probe)
        head = mod(head+1, nb)
        xs[head+1]=x; ys[head+1]=y
        if k ≥ NTRANS*per && (k % per == 0)
            acc += x^2; cnt += 1
        end
    end
    (acc, cnt)
end

println("MC (2-DOF): $(NPATH) paths, dt=$(round(dtm,sigdigits=3)), $(NTRANS)+$(NAVG) periods ...")
t0=time()
nchunk = 8*Threads.nthreads()
chunks = [i:min(i+cld(NPATH,nchunk)-1, NPATH) for i in 1:cld(NPATH,nchunk):NPATH]
accs = zeros(length(chunks)); cnts = zeros(Int, length(chunks))
Threads.@threads for ci in eachindex(chunks)
    a=0.0; c=0
    for i in chunks[ci]
        ai,cci = one_path(2000+i); a+=ai; c+=cci
    end
    accs[ci]=a; cnts[ci]=c
end
var_mc = sum(accs)/sum(cnts)
sem = var_mc*sqrt(2/sum(cnts))
@printf("MC (2-DOF):       Var(x) at period start = %.6e ± %.1e   (%.0f s)\n", var_mc, sem, time()-t0)
@printf("relative deviation SSDM vs MC: %.2f%%  (MC 95%% CI ±%.2f%%)\n",
        100*abs(var_ssdm-var_mc)/var_mc, 100*1.96*sem/var_mc)
