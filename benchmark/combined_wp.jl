# Combined work-precision diagram: local-CPU + local-GPU (+ Colab-GPU when
# benchmark/colab_cpu_gpu.csv exists — paste it from the Colab notebook's
# final cell). All errors are measured against the rho_ref recorded in
# benchmark/cpu_vs_gpu_wp.csv.
#
# Run:  julia --project=. benchmark/combined_wp.jl
# Out:  benchmark/combined_wp.png
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, Printf

local_csv = joinpath(@__DIR__, "cpu_vs_gpu_wp.csv")
colab_csv = joinpath(@__DIR__, "colab_cpu_gpu.csv")

lines = readlines(local_csv)
ρ_ref = parse(Float64, split(lines[1], ":")[2])
rows = NamedTuple[]
for l in lines[3:end]
    f = split(l, ",")
    push!(rows, (p=parse(Int,f[1]), t_cpu=parse(Float64,f[2]),
                 t_gpu=parse(Float64,f[3]), ρ=parse(Float64,f[4])))
end
@printf("local: %d points, ρ_ref=%.12f\n", length(rows), ρ_ref)

plt = plot(title="Work-precision: local CPU / local GPU / Colab GPU\nfully periodic delay-stoch. Mathieu (d=2, SDM q=2)",
           xlabel="wall time [s]", ylabel="|ρ(H) − ρ_ref|",
           xscale=:log10, yscale=:log10, legend=:topright, size=(950,650))
errs = [max(abs(r.ρ-ρ_ref),1e-13) for r in rows]
plot!(plt, [max(r.t_cpu,1e-4) for r in rows], errs, marker=:circle,
      label="local CPU (Xeon Gold 6154, opt.)")
plot!(plt, [max(r.t_gpu,1e-4) for r in rows], errs, marker=:star5,
      label="local GPU (Quadro P4000)")

if isfile(colab_csv)
    clines = readlines(colab_csv)
    gpuname = startswith(clines[1],"#") ? strip(split(clines[1],":")[2]) : "Colab GPU"
    i0 = startswith(clines[1],"#") ? 3 : 2
    crows = NamedTuple[]
    for l in clines[i0:end]
        isempty(strip(l)) && continue
        f = split(l, ",")
        push!(crows, (p=parse(Int,f[1]), t_gpu=parse(Float64,f[3]),
                      ρ=parse(Float64,f[4])))
    end
    cerrs = [max(abs(r.ρ-ρ_ref),1e-13) for r in crows]
    plot!(plt, [r.t_gpu for r in crows], cerrs, marker=:diamond,
          label="Colab GPU ($gpuname)")
    # consistency check: ρ must agree at shared p
    for cr in crows
        i = findfirst(r->r.p==cr.p, rows)
        i === nothing && continue
        dis = abs(cr.ρ-rows[i].ρ)/abs(rows[i].ρ)
        dis > 1e-8 && @warn "ρ mismatch local vs colab" p=cr.p dis
    end
    println("colab: $(length(crows)) points ($gpuname)")
else
    println("(no colab CSV yet — run benchmark/colab_cpu_gpu.ipynb and save the")
    println(" printed CSV block as benchmark/colab_cpu_gpu.csv, then rerun me)")
end
savefig(plt, joinpath(@__DIR__, "combined_wp.png"))
println("wrote benchmark/combined_wp.png")
