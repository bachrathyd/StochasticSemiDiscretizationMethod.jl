# =============================================================================
# FIXPOINT (stationary 2nd-moment) error–resolution diagram, companion to
# grand_orders_pub.jl. Same 12 methods, but now the quantity of interest is the
# stationary variance Var(q) = C*[1,1] of the STABLE PD-Mathieu problem with
# additive (state-independent) noise:
#     ζ=0.2, δ=1, ε=0.5, kP0=0.15, κP=0.3, kD0=0.10, κD=0.4,
#     α=0.35, β=0.25, σ0=0.3   →   ρ(H) ≈ 0.7727  (a genuine fixpoint exists).
# Reference Var(q) = 0.18675690782 (GL5-IBP and GL4-IBP self-converge and agree
# to ~1e-11; the classical SDM path converges to the same value from below).
# Each method records (p, |Var−ref|, CPU time). Live PNG re-saved every point.
# =============================================================================
using Pkg; Pkg.activate("D:/BD/StochasticSemiDiscretizationMethod.jl/benchmark")
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StochasticSemiDiscretizationMethod: StepV8, _lagr_coefs, _lint, _G8, gl_tab,
      Prob, build_v8m, rho_H_krylov_v8m, fixPoint_v8m
using StaticArrays, LinearAlgebra, Plots, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v8_ibp.jl"))

const VREF  = 0.18675690782
const FLOOR = 5e-11        # Krylov/GMRES linsolve accuracy plateau
const TCAP  = 60.0
const PNG   = joinpath(@__DIR__, "out_grand_orders_fix.png")
const CSV   = joinpath(@__DIR__, "grand_orders_fix.csv")

# --- new-method problem (matrix engine) ---
Afun(t)=[0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]
Bfun(t)=[0.0 0.0; 0.15*(1+0.3cos(2π*t)) 0.10*(1+0.4cos(2π*t))]
αfun(t)=[0.0 0.0; 0.35 0.0]
βfun(t)=[0.0 0.0; 0.25 0.0]
σfun(t)=reshape([0.0, 0.3], 2, 1)
pb = Prob(2,1.0,1.0, Afun, Bfun, αfun, βfun, σfun)
const ROUGH=[2]; const POSMAP=Dict(2=>1)
varq_v8(S,p)  = fixPoint_v8m(build_v8m(pb,S,p))[1,1]
varq_ibp(S,p) = fixPoint_v8m(build_v8ibp(pb,S,p,ROUGH,POSMAP))[1,1]

# --- classical SDM problem (factored path, calculate_additive=true) ---
function lddep_fix()
    AM(t)=@SMatrix [0. 1.; -(1.0+0.5cos(2π*t)) -0.4]
    BM(t)=@SMatrix [0. 0.; 0.15*(1+0.3cos(2π*t)) 0.10*(1+0.4cos(2π*t))]
    aM(t)=@SMatrix [0. 0.; 0.35 0.]
    bM(t)=@SMatrix [0. 0.; 0.25 0.]
    LDDEProblem(ProportionalMX(AM), [DelayMX(1.0,BM)],
        [stCoeffMX(1,ProportionalMX(aM))], [stCoeffMX(1,DelayMX(1.0,bM))],
        Additive(2), [stAdditive(1,Additive(@SVector [0., 0.3]))])
end
varq_sdm(q,p) = fixPointOfMapping_MF_factored(
    SSDM.calculateResults(lddep_fix(), SemiDiscretization(q, 1.0/p), 1.0;
                          n_steps=p, calculate_additive=true))[1]

# sanity
let v = varq_ibp(4,16)
    @printf("reference check: GL4-IBP p=16 Var(q)=%.12f (expect ≈%.11f)\n", v, VREF)
    @assert abs(v-VREF) < 1e-9
end

methods = Vector{Tuple{String,Function,Symbol,Int}}()
push!(methods, ("SDM q=2", p->varq_sdm(2,p), :black, 0))
push!(methods, ("SDM q=4", p->varq_sdm(4,p), :gray40, 0))
for S in 1:5
    push!(methods, ("v8 GL$S",  p->varq_v8(S,p),  :none, S))
    push!(methods, ("IBP GL$S", p->varq_ibp(S,p), :none, S))
end
scol = Dict(1=>:dodgerblue, 2=>:seagreen, 3=>:purple, 4=>:darkorange, 5=>:crimson)
data = Dict(m[1]=>(ps=Int[], errs=Float64[], ts=Float64[]) for m in methods)
stopped = Set{String}()

function redraw()
    plt = plot(title="stationary Var(q) error vs resolution — stable PD-Mathieu + additive noise",
               xlabel="p (steps per period)", ylabel="|Var(q) − Var_ref|",
               xscale=:log10, yscale=:log10, legend=:outerright,
               size=(1150,700), framestyle=:box, dpi=150)
    hline!(plt, [FLOOR], color=:gray, ls=:dot, label="solver floor")
    for (name, col, S) in [(m[1],m[3],m[4]) for m in methods]
        d = data[name]; isempty(d.ps) && continue
        c  = S==0 ? col : scol[S]
        ls = startswith(name,"IBP") ? :dash : :solid
        mk = startswith(name,"SDM") ? :circle : (startswith(name,"IBP") ? :diamond : :utriangle)
        plot!(plt, d.ps, max.(d.errs,1e-13), marker=mk, ls=ls, color=c, label=name)
    end
    savefig(plt, PNG)
    open(CSV,"w") do io
        println(io,"method,p,err,t")
        for (name,_,_,_) in methods, i in eachindex(data[name].ps)
            @printf(io,"%s,%d,%.6e,%.4f\n", name, data[name].ps[i], data[name].errs[i], data[name].ts[i])
        end
    end
end

ps_all = [2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,
          160,192,224,256,320,384,448,512,640,768,896,1024,
          1280,1536,1792,2048,2560,3072,3584,4096]
for p in ps_all
    for (name, f, _, _) in methods
        name in stopped && continue
        t = @elapsed v = try f(p) catch e; @warn "$name p=$p" e; push!(stopped,name); continue; end
        err = abs(v-VREF)
        d = data[name]; push!(d.ps,p); push!(d.errs,err); push!(d.ts,t)
        @printf("p=%5d  %-9s err=%.3e  (%.2fs)%s\n", p, name, err, t, t>TCAP ? "  → cap" : "")
        flush(stdout)
        redraw()
        t > TCAP && push!(stopped, name)
        # keep ~5 points past the floor so the saturation plateau is visible
        if count(e -> e < FLOOR, d.errs) ≥ 5
            push!(stopped, name); println("  → $name saturated on floor"); flush(stdout)
        end
    end
    length(stopped) == length(methods) && break
end

println("\n══ final slopes (last 3 resolved points above floor) ══")
for (name,_,_,_) in methods
    d = data[name]; idx = findall(e -> e > FLOOR, d.errs)
    length(idx) < 3 && continue
    i3 = idx[end-2:end]
    sl = log(d.errs[i3[1]]/d.errs[i3[3]]) / log(d.ps[i3[3]]/d.ps[i3[1]])
    @printf("  %-9s slope ≈ %.2f\n", name, sl)
end
println("done — $PNG")
