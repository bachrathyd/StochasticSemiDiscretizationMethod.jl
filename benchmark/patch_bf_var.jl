# One-shot: halve the var column of ssv2_chart_bf.csv for w>0 rows.
# The pre-fix package double-counted the additive source (structures_result.jl
# sizing bug); Var is exactly linear in the additive covariance, so ÷2 is exact.
# The w=0 row was filled analytically (σa²/4ζ) and is already correct.
using DelimitedFiles, Printf
f = joinpath(@__DIR__, "ssv2_chart_bf.csv")
raw, hdr = readdlm(f, ','; header=true)
cp(f, f * ".prefix_bug_backup"; force=true)
open(f, "w") do io
    println(io, join(hdr, ","))
    for i in axes(raw, 1)
        Ω, w, ρ, v = raw[i,1], raw[i,2], raw[i,3], raw[i,4]
        vv = (w > 0 && v isa Real) ? v/2 : v
        @printf(io, "%.6f,%.5f,%.8f,%.8e\n", Ω, w, ρ, float(vv))
    end
end
println("patched ", f)
