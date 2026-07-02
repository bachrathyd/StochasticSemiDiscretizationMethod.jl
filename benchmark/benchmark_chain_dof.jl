# =============================================================================
# CPU vs GPU scaling with MODEL SIZE: chain of n coupled oscillators with every
# "problematic" component — time-periodic stiffness (parametric excitation),
# time-periodic delayed coupling, and time-periodic present + delayed
# multiplicative noise. State dimension d = 2·nDoF.
#
#   M q̈ + C q̇ + K(t) q = Bd(t) q(t−τ) + [α(t)x + β(t)x(t−τ)] dW
#   K(t): tridiagonal chain stiffness, k(t) = k0(1+ε cos t/2)
#   Bd(t): delayed position feedback on every mass, b(t) = b0(1+0.4 cos t/2)
#   α(t): multiplicative present noise ∝ the stiffness rows
#   β(t): multiplicative delayed noise ∝ the delayed coupling
#
# NOTE d-cap of the current pipeline: the Itô isometry operators are stored as
# d²×d² StaticArrays (d⁴ entries) → practical up to nDoF ≈ 5 (d=10, 100×100).
# nDoF = 10/100 needs the factored (Kronecker) operator refactor — see the
# summary in the output.
#
# Run:  julia -t 36 --project=. benchmark/benchmark_chain_dof.jl
# Out:  benchmark/chain_dof.csv, benchmark/chain_dof.png (updated per point)
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, CUDA, Plots, Printf

CUDA.functional() || error("needs a CUDA GPU")
const τm=2π; const Pm=4π

# chain matrices as plain functions of t (converted to SMatrix at build)
function chain_problem(::Val{n}) where {n}
    d = 2n
    k0=3.0; εk=1.0; ζ=0.05; b0=0.4; αs=0.08
    Kmat(t) = begin
        k = k0*(1+εk*cos(0.5t))
        K = zeros(n,n)
        for i in 1:n
            K[i,i] = 2k
            i>1 && (K[i,i-1] = -k)
            i<n && (K[i,i+1] = -k)
        end
        K
    end
    Afun(t) = begin
        A = zeros(d,d)
        A[1:n, n+1:d] .= Matrix(I,n,n)
        A[n+1:d, 1:n] .= -Kmat(t)
        A[n+1:d, n+1:d] .= -2ζ .* Matrix(I,n,n)
        SMatrix{d,d,Float64}(A)
    end
    Bfun(t) = begin
        B = zeros(d,d)
        b = b0*(1+0.4cos(0.5t))
        for i in 1:n; B[n+i, i] = b; end
        SMatrix{d,d,Float64}(B)
    end
    αfun(t) = begin
        A = zeros(d,d)
        A[n+1:d, 1:n] .= -αs .* Kmat(t)
        SMatrix{d,d,Float64}(A)
    end
    βfun(t) = begin
        B = zeros(d,d)
        b = αs*b0*(1+0.4cos(0.5t))
        for i in 1:n; B[n+i, i] = b; end
        SMatrix{d,d,Float64}(B)
    end
    LDDEProblem(ProportionalMX(Afun), [DelayMX(τm, Bfun)],
                [stCoeffMX(1, ProportionalMX(αfun))],
                [stCoeffMX(1, DelayMX(τm, βfun))],
                Additive(d), [stAdditive(1, Additive(SVector{d,Float64}(zeros(d))))])
end

mapping(lddep, p) = DiscreteMapping_M2_MF(
    StochasticSemiDiscretizationMethod.calculateResults(
        lddep, SemiDiscretization(2, Pm/p), τm, n_steps=p))

timeit(f) = (t0=time(); v=f(); (time()-t0, v))
function timeit3(f); t,v=timeit(f); for _ in 1:2; t2,_=timeit(f); t=min(t,t2); end; (t,v); end

csv_path = joinpath(@__DIR__, "chain_dof.csv")
png_path = joinpath(@__DIR__, "chain_dof.png")
rows = NamedTuple[]

function save_all(rows)
    open(csv_path, "w") do io
        println(io, "ndof,d,p,D,t_cpu,t_gpu,rho_cpu,rho_gpu")
        for r in rows
            @printf(io,"%d,%d,%d,%d,%.4f,%.4f,%.12f,%.12f\n",
                    r.ndof,r.d,r.p,r.D,r.t_cpu,r.t_gpu,r.ρc,r.ρg)
        end
    end
    plt1 = plot(title="CPU vs GPU across model size (chain, all-periodic SDDE)",
                xlabel="p (steps/period)", ylabel="wall time [s]",
                xscale=:log10, yscale=:log10, legend=:topleft)
    plt2 = plot(title="GPU speedup vs p", xlabel="p", ylabel="t_CPU / t_GPU",
                xscale=:log10, legend=:topleft)
    cols = Dict(1=>:blue, 2=>:red, 3=>:green, 5=>:purple)
    for nd in sort(unique(r.ndof for r in rows))
        sel = filter(r->r.ndof==nd, rows); isempty(sel) && continue
        pv=[r.p for r in sel]
        plot!(plt1, pv, [max(r.t_cpu,1e-4) for r in sel], marker=:circle,
              color=get(cols,nd,:black), label="CPU $(nd)DoF (d=$(2nd))")
        plot!(plt1, pv, [max(r.t_gpu,1e-4) for r in sel], marker=:star5, ls=:dash,
              color=get(cols,nd,:black), label="GPU $(nd)DoF")
        plot!(plt2, pv, [r.t_cpu/r.t_gpu for r in sel], marker=:diamond,
              color=get(cols,nd,:black), label="$(nd)DoF (d=$(2nd))")
    end
    hline!(plt2, [1.0], color=:gray, ls=:dot, label="")
    savefig(plot(plt1, plt2, layout=(1,2), size=(1500,600)), png_path)
end

const TIME_CAP = 400.0
for ndof in (1, 2, 3, 5)
    d = 2ndof
    println("═══ $(ndof)-DoF chain (d=$d) ═══"); flush(stdout)
    t_build = @elapsed lddep = chain_problem(Val(ndof))
    # warm
    dmw = mapping(lddep, 8)
    spectralRadiusOfMapping_MF(dmw); spectralRadiusOfMapping_GPU(dmw)
    stopped_cpu = false; stopped_gpu = false
    for p in (16, 24, 32, 48, 64, 96, 128, 192, 256)
        (stopped_cpu && stopped_gpu) && break
        dm = mapping(lddep, p)
        r = div(dm.rst.n, d) - 1
        D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
        t_cpu = NaN; ρc = NaN; t_gpu = NaN; ρg = NaN
        if !stopped_cpu
            t_cpu, ρc = timeit3(() -> spectralRadiusOfMapping_MF(dm))
            t_cpu > TIME_CAP && (stopped_cpu = true)
        end
        if !stopped_gpu
            t_gpu, ρg = timeit3(() -> spectralRadiusOfMapping_GPU(dm))
            t_gpu > TIME_CAP && (stopped_gpu = true)
        end
        dis = (isnan(ρc)||isnan(ρg)) ? NaN : abs(ρc-ρg)/abs(ρc)
        push!(rows, (ndof=ndof, d=d, p=p, D=D, t_cpu=t_cpu, t_gpu=t_gpu, ρc=ρc, ρg=ρg))
        @printf("  p=%4d D=%9d CPU %8.3fs GPU %8.3fs speedup %6.2f× mismatch %.1e %s\n",
                p, D, t_cpu, t_gpu, t_cpu/t_gpu, dis,
                (isnan(dis) || dis < 1e-8) ? "OK" : "MISMATCH!")
        flush(stdout)
        save_all(rows)                    # live figure after every point
    end
end
println("done — $csv_path, $png_path")
