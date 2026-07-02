# =============================================================================
# v8 MATRIX engine — validation ladder, stage 1
#   M-1 deterministic: scalar Hayes exact multiplier 0.3319869969
#   M-2 noise-off gates, d=1 and d=2 (ρH = ρU², exact)
#   M-3 present-noise exact value exp((2a+α²)T)
#   M-4 scalar regressions vs arbitrated refs (v8-scalar fixed these):
#         a: B=−.4, β=.3 → 0.1473709451
#         c: B=−.4, α=.3 → 0.0868082230
#         d: B=−.4, α=β=.3 → 0.1701952437
#   M-5 case e (B=0, α=.3, β=.3): αβ-cross WITHOUT delay drift — isolates
#       whether case d's O(h³) tail needs B or only the cross term.
# =============================================================================
include(joinpath(@__DIR__, "cov_colloc_v8.jl"))
using Printf

fails=Ref(0)
gate(name, ok) = (fails[] += ok ? 0 : 1; @printf("%-64s %s\n", name, ok ? "PASS" : "FAIL"))

println("── M-1 deterministic: Hayes exact μ = 0.3319869969 ──")
pbdet = Prob(1, 1.0, 1.0, t->fill(-1.0,1,1), t->fill(-0.5,1,1),
             t->zeros(1,1), t->zeros(1,1))
for S in (2,3)
    errs=Float64[]; ps=[4,8,16,32]
    for p in ps
        push!(errs, abs(rho_U_v8m(build_v8m(pbdet,S,p)) - 0.3319869969))
    end
    rates=[log2(errs[i]/errs[i+1]) for i in 1:3]
    @printf("  GL%d errs=%s rates=%s\n", S,
            join([@sprintf("%.2e",e) for e in errs],","),
            join([@sprintf("%.2f",r) for r in rates],","))
    gate("  GL$S deterministic order", errs[end]<1e-11 || rates[end]>2S-0.7)
end

println("── M-2 noise-off gates ──")
for (nm, pb, S, p) in (("d=1 Hayes", pbdet, 2, 8),
                       ("d=2 mirror", Prob(2,1.0,1.0,
                            t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
                            t->[0.0 0.0; 0.2 0.0],
                            t->zeros(2,2), t->zeros(2,2)), 2, 8))
    eng=build_v8m(pb,S,p)
    ρU2=rho_U_v8m(eng)^2
    ρH=rho_H_krylov_v8m(eng)
    rel=abs(ρH-ρU2)/ρU2
    gate(@sprintf("  %s: ρH=%.10f ρU²=%.10f rel=%.1e", nm, ρH, ρU2, rel), rel<1e-10)
end

println("── M-3 present-noise exact exp((2a+α²)T) ──")
ρex=exp(2*(-0.8)+0.16)
pbc = Prob(1, 1.0, 0.25, t->fill(-0.8,1,1), t->zeros(1,1),
           t->fill(0.4,1,1), t->zeros(1,1))
for S in (2,3)
    errs=Float64[]; ps=[4,8,16,32]
    for p in ps
        push!(errs, abs(rho_H_krylov_v8m(build_v8m(pbc,S,p)) - ρex))
    end
    rates=[log2(errs[i]/errs[i+1]) for i in 1:3]
    @printf("  GL%d errs=%s rates=%s\n", S,
            join([@sprintf("%.2e",e) for e in errs],","),
            join([@sprintf("%.2f",r) for r in rates],","))
    gate("  GL$S present-noise order", errs[end]<1e-12 || rates[end]>2S-0.7)
end

println("── M-4/M-5 scalar cases ──")
cases = [
    ("a: B=−.4 β=.3      ", -0.4, 0.3, 0.0, 0.1473709451),
    ("c: B=−.4 α=.3      ", -0.4, 0.0, 0.3, 0.0868082230),
    ("d: B=−.4 β=.3 α=.3 ", -0.4, 0.3, 0.3, 0.1701952437),
    ("e: B=0   β=.3 α=.3 ",  0.0, 0.3, 0.3, NaN),   # ref from own GL3 fine
]
ps=[4,6,8,12,16,24,32]
for (name,Bv,βv,αv,ρref) in cases
    pb = Prob(1, 1.0, 1.0, t->fill(-1.0,1,1), t->fill(Bv,1,1),
              t->fill(αv,1,1), t->fill(βv,1,1))
    ref = isnan(ρref) ? rho_H_krylov_v8m(build_v8m(pb,3,48)) : ρref
    println("  case $name ref=$(round(ref,digits=10))")
    for S in (2,3)
        errs=Float64[]
        for p in ps
            push!(errs, abs(rho_H_krylov_v8m(build_v8m(pb,S,p)) - ref))
        end
        rates=[log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
        @printf("    GL%d errs: %s\n", S, join([@sprintf("%.2e",e) for e in errs]," "))
        @printf("    GL%d rates: %s\n", S, join([@sprintf("%5.2f",r) for r in rates]," "))
        flush(stdout)
    end
end
println(fails[]==0 ? "ALL STAGE-1 GATES PASS" : "$(fails[]) GATE FAILURES")
