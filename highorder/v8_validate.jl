# =============================================================================
# v8 scalar prototype — validation ladder (V8_DESIGN.md items 1–4)
#   V8-1 deterministic order: scalar Hayes exact multiplier 0.3319869969
#   V8-2 noise-off gate ρ(H)=ρ(U)²  (scalar → real dominant mode, Krylov fine)
#   V8-3 present-noise no-delay exact value exp((2a+α²)T)
#   V8-4 THE TARGET: cases a/c/d (arbitrated refs) must lift beyond v7's O(h²)
# =============================================================================
include(joinpath(@__DIR__, "cov_colloc_v8_scalar.jl"))
using Printf

println("── V8-1 deterministic: Hayes exact μ = 0.3319869969 ──")
pbdet = Prob(1, 1.0, 1.0, t->fill(-1.0,1,1), t->fill(-0.5,1,1),
             t->zeros(1,1), t->zeros(1,1))
for S in (2,3)
    errs=Float64[]; ps=[4,8,16,32]
    for p in ps
        push!(errs, abs(rho_U_v8(build_v8(pbdet,S,p)) - 0.3319869969))
    end
    rates=[log2(errs[i]/errs[i+1]) for i in 1:length(errs)-1]
    @printf("  GL%d errs=%s rates=%s\n", S,
            join([@sprintf("%.2e",e) for e in errs],","),
            join([@sprintf("%.2f",r) for r in rates],","))
end

println("── V8-2 noise-off gate (B=−0.5, α=β=0) ──")
for S in (2,3), p in (6,10)
    eng=build_v8(pbdet,S,p)
    ρU2=rho_U_v8(eng)^2
    ρH=rho_H_krylov_v8(eng)
    rel=abs(ρH-ρU2)/ρU2
    @printf("  GL%d p=%d ρH=%.12f ρU²=%.12f rel=%.1e %s\n", S,p,ρH,ρU2,rel,
            rel<1e-10 ? "PASS" : "FAIL")
end

println("── V8-3 present-noise exact: exp((2a+α²)T), a=−0.8, α=0.4 ──")
ρex=exp(2*(-0.8)+0.16)
pbc = Prob(1, 1.0, 0.25, t->fill(-0.8,1,1), t->zeros(1,1),
           t->fill(0.4,1,1), t->zeros(1,1))
for S in (2,3)
    errs=Float64[]; ps=[4,8,16,32]
    for p in ps
        push!(errs, abs(rho_H_krylov_v8(build_v8(pbc,S,p)) - ρex))
    end
    rates=[log2(errs[i]/errs[i+1]) for i in 1:length(errs)-1]
    @printf("  GL%d errs=%s rates=%s\n", S,
            join([@sprintf("%.2e",e) for e in errs],","),
            join([@sprintf("%.2f",r) for r in rates],","))
end

println("── V8-4 TARGET cases (arbitrated refs; v7 measured O(h²) on all) ──")
cases = [
    ("a: B=−.4 β=.3     ", -0.4, 0.3, 0.0, 0.1473709451),
    ("c: B=−.4 α=.3     ", -0.4, 0.0, 0.3, 0.0868082230),
    ("d: B=−.4 β=.3 α=.3", -0.4, 0.3, 0.3, 0.1701952437),
]
ps=[4,6,8,12,16,24,32]
for (name,Bv,βv,αv,ρref) in cases
    println("  case $name  ref=$ρref")
    pb = Prob(1, 1.0, 1.0, t->fill(-1.0,1,1), t->fill(Bv,1,1),
              t->fill(αv,1,1), t->fill(βv,1,1))
    for S in (2,3)
        errs=Float64[]
        for p in ps
            push!(errs, abs(rho_H_krylov_v8(build_v8(pb,S,p)) - ρref))
        end
        rates=[log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
        @printf("    GL%d errs: %s\n", S, join([@sprintf("%.2e",e) for e in errs]," "))
        @printf("    GL%d rates: %s\n", S, join([@sprintf("%5.2f",r) for r in rates]," "))
        flush(stdout)
    end
end
println("done")
