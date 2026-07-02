# =============================================================================
# v7 validation ladder — stage 1: structural gates (no package reference needed)
#
# G-A  noise-off gate: α=β=0, B≠0, periodic A(t)  ⇒  ρ(H) = ρ(U)² to ~1e-13,
#      in BOTH offdiag modes (:none = v6 behavior, :causal = v7 fill).
# G-B  Krylov vs dense ρ(H) agreement on a small case (both modes).
# G-C  present-noise no-delay exact value: dx = a x dt + α x dW over T ⇒
#      ρ(H) = exp((2a+α²)T); order 2S for GL1/2/3 (both modes — B=β=0 so the
#      fill should be inert; verifies the fill breaks nothing).
# =============================================================================
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))
using Printf

fails = Ref(0)
function gate(name, ok)
    @printf("%-58s %s\n", name, ok ? "PASS" : "FAIL")
    ok || (fails[] += 1)
end

# ---------- G-A: noise-off gate (Mathieu drift, B≠0) -------------------------
println("── G-A: noise-off gate (B=0.2, α=β=0, periodic A) ──")
pbA = Prob(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.2 0.0],
    t->zeros(2,2), t->zeros(2,2))
for S in (2,3), p in (6, 10)
    eng = build_v7(pbA, S, p)
    ρU2 = rho_U_v7(eng)^2
    for mode in (:none, :causal)
        ρH = rho_H_krylov(eng; offdiag=mode)
        rel = abs(ρH-ρU2)/ρU2
        gate(@sprintf("  GL%d p=%d %-7s ρH=%.12f ρU²=%.12f rel=%.1e", S,p,mode,ρH,ρU2,rel), rel < 1e-11)
    end
end

# ---------- G-B: Krylov vs dense -------------------------------------------
println("── G-B: Krylov vs dense ρ(H) (small case, with noise) ──")
pbB = Prob(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.2 0.0],
    t->[0.0 0.0; 0.2 0.0],
    t->[0.0 0.0; 0.2 0.0])
for mode in (:none, :causal)
    eng = build_v7(pbB, 2, 6)
    ρd = rho_H_dense(eng; offdiag=mode)
    ρk = rho_H_krylov(eng; offdiag=mode)
    rel = abs(ρd-ρk)/abs(ρd)
    gate(@sprintf("  GL2 p=6 %-7s dense=%.12f krylov=%.12f rel=%.1e", mode,ρd,ρk,rel), rel < 1e-9)
end

# ---------- G-C: present-noise exact value + order ---------------------------
println("── G-C: present-only noise, no delay coupling: ρ = exp((2a+α²)T) ──")
a_c = -0.8; α_c = 0.4; T_c = 1.0
ρ_exact = exp((2a_c + α_c^2)*T_c)
@printf("  exact ρ = %.12f\n", ρ_exact)
pbC = Prob(1, T_c, 0.25,
    t->fill(a_c,1,1), t->zeros(1,1), t->fill(α_c,1,1), t->zeros(1,1))
for mode in (:none, :causal)
    println("  mode = $mode")
    for S in (1,2,3)
        errs=Float64[]; ps=[4,8,16,32]
        for p in ps
            eng = build_v7(pbC, S, p)
            ρ = rho_H_krylov(eng; offdiag=mode)
            push!(errs, abs(ρ-ρ_exact))
        end
        rates=[log2(errs[i]/errs[i+1]) for i in 1:length(errs)-1]
        @printf("    GL%d errs=%s rates=%s\n", S,
                join([@sprintf("%.2e",e) for e in errs],","),
                join([@sprintf("%.2f",r) for r in rates],","))
        # gate: measured rate at the finest pair within 0.7 of nominal 2S,
        # or already at the accuracy floor (err < 1e-12)
        ok = errs[end] < 1e-12 || rates[end] > 2S - 0.7
        gate(@sprintf("    GL%d order (nominal %d)", S, 2S), ok)
    end
end

println(fails[]==0 ? "ALL GATES PASS" : "$(fails[]) GATE FAILURES")
exit(fails[]==0 ? 0 : 1)
