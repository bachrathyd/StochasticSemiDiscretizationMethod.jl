# Additional WP baselines requested in review (Reviewer C5/D6):
#   (1) "recursion (copy)" — the natural intermediate classical formulation:
#       the p sparse single-step second-moment matrices are applied in sequence
#       inside each Krylov operator application (no explicit product, no MF
#       buffer machinery). Cost O(p · nnz) = O(p^3) per apply, O(p^2) memory.
#   (2) Krylov iteration counts (numops) for this baseline AND the MF path,
#       to substantiate the p-independence of the iteration count.
# Same problem/reference as benchmark_wp_ultra.jl. Output: wp_baseline.csv
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, KrylovKit
BLAS.set_num_threads(1)

function lddep()
    AM(t)=@SMatrix [0. 1.; -(1.0+0.8cos(2π*t)) -(0.10+0.02cos(2π*t))]
    BM(t)=@SMatrix [0. 0.; 0.35*(1+0.3cos(2π*t)) 0.]
    aM(t)=@SMatrix [0. 0.; 0.45*(1+0.2cos(2π*t)) 0.]
    bM(t)=@SMatrix [0. 0.; 0.30*(1+0.4cos(2π*t)) 0.]
    LDDEProblem(ProportionalMX(AM), [DelayMX(1.0,BM)],
        [stCoeffMX(1,ProportionalMX(aM))], [stCoeffMX(1,DelayMX(1.0,bM))],
        Additive(2), [stAdditive(1,Additive(@SVector [0.,0.]))])
end
rst_at(p) = SSDM.calculateResults(lddep(), SemiDiscretization(2, 1.0/p), 1.0, n_steps=p)

println("computing reference ..."); flush(stdout)
ρ1 = spectralRadiusOfMapping_MF_factored(rst_at(1024); krylovdim=15)
ρ2 = spectralRadiusOfMapping_MF_factored(rst_at(2048); krylovdim=15)
const ρref = 2ρ2 - ρ1
@printf("ρref=%.10f\n", ρref); flush(stdout)

# copy-recursion spectral radius with iteration count
function rho_recursion(rst)
    dm = DiscreteMapping_M2(rst)             # p sparse step matrices (vech)
    MXs = dm.M2_MXs
    D = size(MXs[1], 2)
    op(v) = foldl((x,M)->M*x, MXs; init=v)
    vals, _, info = eigsolve(op, rand(D), 1, :LM; krylovdim=15, maxiter=100)
    (abs(vals[1]), info.numops, D)
end
# MF factored with iteration count (mirrors spectralRadiusOfMapping_MF_factored)
function rho_mf(rst, d)
    r = div(rst.n, d) - 1
    D = SSDM.CovVecIdx((r+1)*d).sectionStarts[end]
    cf = SSDM.get_factored_coefficients(rst; include_additive=false)
    ws = SSDM.MFFactoredWorkspace(d, r)
    op = SSDM.MFFactoredOperator(cf, rst, D, ws)
    vals, _, info = eigsolve(op, rand(D), 1, :LM; krylovdim=15, maxiter=100)
    (abs(vals[1]), info.numops)
end

const TCAP = 300.0
open(joinpath(@__DIR__,"wp_baseline.csv"),"w") do io
    println(io,"method,p,D,time_s,mem_MB,err,numops")
    stopped_rec = false
    for p in [8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048]
        rst = rst_at(p)
        GC.gc()
        st = @timed rho_mf(rst, 2)
        ρ, nops = st.value
        @printf(io,"MF,%d,%d,%.4f,%.2f,%.6e,%d\n", p,
                SSDM.CovVecIdx(rst.n).sectionStarts[end], st.time, st.bytes/1e6,
                abs(ρ-ρref), nops)
        @printf("p=%5d MF        t=%8.2fs numops=%d err=%.3e\n", p, st.time, nops,
                abs(ρ-ρref)); flush(stdout); flush(io)
        if !stopped_rec
            GC.gc()
            st2 = try @timed rho_recursion(rst) catch e; @warn e; stopped_rec=true; nothing end
            if st2 !== nothing
                ρ2v, nops2, D2 = st2.value
                @printf(io,"recursion,%d,%d,%.4f,%.2f,%.6e,%d\n", p, D2, st2.time,
                        st2.bytes/1e6, abs(ρ2v-ρref), nops2)
                @printf("p=%5d recursion t=%8.2fs numops=%d err=%.3e%s\n", p, st2.time,
                        nops2, abs(ρ2v-ρref), st2.time>TCAP ? " → cap" : "")
                flush(stdout); flush(io)
                st2.time > TCAP && (stopped_rec = true)
            end
        end
    end
end
println("done — wp_baseline.csv")
