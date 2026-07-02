# =============================================================================
# IBP proof-of-concept test. Delayed-VELOCITY-feedback stochastic Mathieu:
#   q̈ + 2ζq̇ + k(t)q = b_D(t) q̇(t−τ) + [α q + β q(t−τ)] dW      (pure D)
#   q̈ + 2ζq̇ + k(t)q = b_P q(t−τ) + b_D q̇(t−τ) + [α q + β q(t−τ)]dW  (PD)
# State [q,v], d=2, rough_cols=[2] (velocity), posmap 2→1.
#
# Gates:
#  G1 noise-off: ρ(H)=ρ(U)² exact, both v8 and v8-IBP (IBP must not break it)
#  G2 PURE-D order: v8 direct caps ~O(h⁴); v8-IBP → O(h^2S). Ref = fine arbiter.
#  G3 PD order: same, with position feedback added.
# =============================================================================
include(joinpath(@__DIR__, "cov_colloc_v8_ibp.jl"))
include(joinpath(@__DIR__, "arbiter_finegrid.jl"))
using Printf

const ROUGH = [2]; const POSMAP = Dict(2=>1)

# ── G1: noise-off gate (b_D velocity feedback, α=β=0) ──
println("── G1 noise-off gate (delayed velocity feedback, α=β=0) ──")
pb_off = Prob(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.3cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.0 0.15*(1+0.3cos(2π*t))],   # b_D on velocity
    t->zeros(2,2), t->zeros(2,2))
for S in (2,3), p in (6,10)
    eng = build_v8ibp(pb_off, S, p, ROUGH, POSMAP)
    ρU2 = rho_U_v8m(eng)^2
    ρH  = rho_H_krylov_v8m(eng)
    rel = abs(ρH-ρU2)/ρU2
    @printf("  GL%d p=%d ρH=%.12f ρU²=%.12f rel=%.1e %s\n", S,p,ρH,ρU2,rel, rel<1e-10 ? "PASS" : "FAIL")
end

function order(engf, ref, ps, S)
    errs=[abs(engf(S,p)-ref) for p in ps]
    rates=[log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
    (errs, rates)
end

# ── G2: PURE delayed-velocity feedback ──
println("── G2 pure delayed-velocity Mathieu (b_D only) ──")
Afun(t)=[0.0 1.0; -(1.0+0.3cos(2π*t)) -0.2]
BfunD(t)=[0.0 0.0; 0.0 0.15*(1+0.3cos(2π*t))]
αfun(t)=[0.0 0.0; 0.25 0.0]; βfun(t)=[0.0 0.0; 0.15 0.0]
pb_v8  = Prob(2,1.0,1.0, Afun, BfunD, αfun, βfun)
pb_fg  = FGProb(2,1.0,1.0, Afun, BfunD, αfun, βfun)
println("  arbiter reference:")
ρref = fg_arbiter(pb_fg, [512,1024,2048])
ps=[6,8,12,16,24,32]
ed,rd = order((S,p)->rho_H_krylov_v8m(build_v8m(pb_v8,S,p)), ρref, ps, 3)   # direct v8
ei,ri = order((S,p)->rho_H_krylov_v8m(build_v8ibp(pb_v8,S,p,ROUGH,POSMAP)), ρref, ps, 3) # IBP
@printf("  v8-direct GL3 errs: %s\n", join([@sprintf("%.2e",e) for e in ed]," "))
@printf("  v8-direct GL3 rates: %s\n", join([@sprintf("%.2f",r) for r in rd]," "))
@printf("  v8-IBP    GL3 errs: %s\n", join([@sprintf("%.2e",e) for e in ei]," "))
@printf("  v8-IBP    GL3 rates: %s  (target ≥6)\n", join([@sprintf("%.2f",r) for r in ri]," "))
# GL2 too
ed2,rd2 = order((S,p)->rho_H_krylov_v8m(build_v8m(pb_v8,S,p)), ρref, ps, 2)
ei2,ri2 = order((S,p)->rho_H_krylov_v8m(build_v8ibp(pb_v8,S,p,ROUGH,POSMAP)), ρref, ps, 2)
@printf("  v8-direct GL2 rates: %s\n", join([@sprintf("%.2f",r) for r in rd2]," "))
@printf("  v8-IBP    GL2 rates: %s  (target ≥4)\n", join([@sprintf("%.2f",r) for r in ri2]," "))

# ── G3: full PD ──
println("── G3 full PD (position + velocity feedback) ──")
BfunPD(t)=[0.0 0.0; 0.3*(1+0.2cos(2π*t)) 0.15*(1+0.3cos(2π*t))]
pb_pd  = Prob(2,1.0,1.0, Afun, BfunPD, αfun, βfun)
pb_pdfg= FGProb(2,1.0,1.0, Afun, BfunPD, αfun, βfun)
println("  arbiter reference:")
ρref2 = fg_arbiter(pb_pdfg, [512,1024,2048])
edp,rdp = order((S,p)->rho_H_krylov_v8m(build_v8m(pb_pd,S,p)), ρref2, ps, 3)
eip,rip = order((S,p)->rho_H_krylov_v8m(build_v8ibp(pb_pd,S,p,ROUGH,POSMAP)), ρref2, ps, 3)
@printf("  v8-direct GL3 rates: %s\n", join([@sprintf("%.2f",r) for r in rdp]," "))
@printf("  v8-IBP    GL3 rates: %s  (target ≥6)\n", join([@sprintf("%.2f",r) for r in rip]," "))
println("done")
