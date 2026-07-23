# =============================================================================
# High-order collocation on SSV MILLING: the engagement-regularity order ladder.
#
# The 2-DOF SSV milling model of Fig. `fig:ssv` uses an INTERRUPTED (straight-
# tooth) cut, whose directional matrix H(t) JUMPS as teeth cross the radial
# engagement boundary. Those jumps drift across the (SSV-modulated) time grid and
# fall inside collocation steps, so the Gauss–Legendre stage interpolation
# Gibbs-oscillates and the second-moment spectral radius does NOT converge — which
# is why that chart is (correctly) computed with the classical scheme.
#
# This benchmark shows the flip side: the moment collocation recovers its high
# order the moment the engagement is smoothed, as a real HELICAL tool does. A
# helical flute averages the local cutting force over the axial depth, i.e. over
# an angular helix-lag window ψ — turning the hard on/off screen into a continuous
# (C⁰, trapezoidal) engaged fraction. Smoothing the screen corners once more (a C²
# axial weighting) removes the remaining kinks. Measured S=3 order climbs with the
# coefficient regularity:  interrupted C⁻¹ → ~0 (no convergence),  helical C⁰ → ~3,
# smoothed C² → ~5 (approaching 2S=6). The classical scheme stays first order
# throughout (shown as the baseline).
#
# Outputs (incremental — CSV + re-render after every point):
#   benchmark/helical_milling_order.csv
#   assets/HelicalMillingOrder.png
#   journal_paper/images/helical_milling_order.{png,pdf}   (if the dir exists)
# =============================================================================
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, DelimitedFiles, Plots
BLAS.set_num_threads(1)

const N_TEETH=2; const PHI_EX=π/4; const Kr=0.3; const RVA=0.25; const NT=10; const ζ=0.02
const PSI=π/4                       # helix-lag window (= immersion ⇒ strongly helical)
const Ω0=1.0; const W=0.25          # representative STABLE operating point (ρ≈0.3–0.6)
const σc=0.20; const σa=0.10

φ0fun(t,Ω0,Tssv,rva)= rva==0 ? Ω0*t : Ω0*t-(Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv)-1.0)
@inline function Mloc(φ)                        # local force-direction matrix (smooth in φ)
    s,c=sincos(φ); a1=(c+Kr*s); a2=(s-Kr*c)
    SMatrix{2,2,Float64}(a1*s, -a2*s, a1*c, -a2*c)
end
@inline smoo(x)= x≤0 ? 0.0 : x≥1 ? 1.0 : x^3*(x*(x*6-15)+10)     # C² quintic smootherstep
@inline function gfrac(φ, mode)                 # engaged fraction of the helix window
    mode==:hard && return (0.0≤φ≤PHI_EX) ? 1.0 : 0.0
    lo=max(φ-PSI,0.0); hi=min(φ,PHI_EX); ov=max(0.0,hi-lo)/PSI
    mode==:smooth ? smoo(ov) : clamp(ov,0.0,1.0)                 # :helical ⇒ C⁰ trapezoid
end
function Hgen(t,Tssv,mode)
    φ0=φ0fun(t,Ω0,Tssv,RVA); H=@MMatrix zeros(2,2)
    for j in 0:N_TEETH-1
        φ=mod(φ0+2π*j/N_TEETH,2π); g=gfrac(φ,mode); g==0 && continue
        H .+= g.*Mloc(φ)
    end
    SMatrix{2,2}(H)
end
function mkprob(mode)
    Tssv=NT*2π/Ω0
    τf(t)=(2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
    τmax=(2π/N_TEETH)/(Ω0*(1-RVA))
    Hf(t)=Hgen(t,Tssv,mode)
    Af(t)=(H=Hf(t); SMatrix{4,4,Float64}(0.,0.,-1-W*H[1,1],-W*H[2,1], 0.,0.,-W*H[1,2],-1-W*H[2,2], 1.,0.,-2ζ,0., 0.,1.,0.,-2ζ))
    Bf(t)=(H=Hf(t); SMatrix{4,4,Float64}(0.,0.,W*H[1,1],W*H[2,1], 0.,0.,W*H[1,2],W*H[2,2], 0.,0.,0.,0., 0.,0.,0.,0.))
    af(t)=(H=Hf(t); SMatrix{4,4,Float64}(0.,0.,-σc*W*H[1,1],-σc*W*H[2,1], 0.,0.,-σc*W*H[1,2],-σc*W*H[2,2], 0.,0.,0.,0., 0.,0.,0.,0.))
    bf(t)=(H=Hf(t); SMatrix{4,4,Float64}(0.,0.,σc*W*H[1,1],σc*W*H[2,1], 0.,0.,σc*W*H[1,2],σc*W*H[2,2], 0.,0.,0.,0., 0.,0.,0.,0.))
    lddep=LDDEProblem(ProportionalMX(Af),[DelayMX(τf,Bf)],
        [stCoeffMX(1,ProportionalMX(af))],[stCoeffMX(1,DelayMX(τf,bf))],
        Additive(4),[stAdditive(2,Additive(@SVector [0.,0.,σa,0.]))])
    (lddep,Tssv,τmax)
end
function ρcoll(mode,p;S=3)
    lddep,Tssv,_=mkprob(mode)
    spectralRadiusOfMoment(lddep,Tssv,p;method=Collocation(S),verbosity=0)
end
function ρclass(mode;R_RES=24,NAT_RES=30)
    lddep,Tssv,τmax=mkprob(mode)
    Δt=min(τmax/R_RES,2π/NAT_RES); nst=max(1,round(Int,Tssv/Δt))
    spectralRadiusOfMapping_MF_factored(SSDM.calculateResults(lddep,SemiDiscretization(2,Tssv/nst),τmax;n_steps=nst))
end

const CSV=joinpath(@__DIR__,"helical_milling_order.csv")
const ASSET=joinpath(@__DIR__,"..","assets","HelicalMillingOrder.png")
const PAPER=joinpath(@__DIR__,"..","journal_paper","images")
const PS=(48,64,96,128,192,256,384)
const PREF=1536
# per-model reference: convergent models self-reference to a fine collocation value;
# the non-convergent interrupted cut is referenced to the trustworthy classical value.
const MODES=(:hard,:helical,:smooth)
have=Dict{Tuple{Symbol,Int},Float64}()
if isfile(CSV)
    raw,_=readdlm(CSV,',';header=true)
    for k in axes(raw,1); have[(Symbol(raw[k,1]),Int(raw[k,2]))]=Float64(raw[k,3]); end
end
# references are cached in the CSV as (mode, PREF) rows so the err column can
# never go stale against a later ref recompute (the convergent modes self-reference
# to a fine collocation value; the non-convergent interrupted cut to classical).
getref!(m, f) = haskey(have,(m,PREF)) ? have[(m,PREF)] : (have[(m,PREF)]=f())
refs=Dict(:helical=>getref!(:helical, ()->ρcoll(:helical,PREF)),
          :smooth=>getref!(:smooth, ()->ρcoll(:smooth,PREF)),
          :hard=>getref!(:hard, ()->ρclass(:hard)))
save()=open(CSV,"w") do io
    println(io,"mode,p,rho,ref,err")
    for m in MODES, p in vcat(collect(PS), PREF)
        haskey(have,(m,p)) || continue
        ρ=have[(m,p)]; println(io,"$m,$p,$ρ,$(refs[m]),$(abs(ρ-refs[m]))")
    end
end
save()
STY=Dict(:hard=>("interrupted cut  (C^-1, discontinuous)",:black,:xcross,:dash),
         :helical=>("helical  (C^0)",:royalblue,:diamond,:solid),
         :smooth=>("smoothed screen  (C^2)",:firebrick,:circle,:solid))
function render()
    plt=plot(size=(760,560),dpi=300,framestyle=:box,legend=:bottomleft,
             xscale=:log10,yscale=:log10,xlabel="steps per period  p",
             ylabel="error in ρ(H)",guidefontsize=12,tickfontsize=10,
             title="SSV milling: collocation order vs engagement regularity (S=3)",
             left_margin=5Plots.mm,bottom_margin=5Plots.mm)
    for m in MODES
        pts=sort([(p,abs(have[(m,p)]-refs[m])) for p in PS if haskey(have,(m,p))])
        isempty(pts) && continue
        xs=[p for (p,_) in pts]; es=[max(e,1e-16) for (_,e) in pts]
        name,col,mk,ls=STY[m]
        # LSQ slope over the LONGEST strictly-decreasing run (drifting C^k corners
        # cause isolated bounces; the fine-p tail sits at the reference floor)
        if m==:hard
            name*="  (non-convergent)"
        else
            bs,be,s0=1,1,1
            for i in 2:length(es)
                es[i]<es[i-1] || (s0=i)
                i-s0>be-bs && (bs=s0; be=i)
            end
            if be-bs+1≥3
                lx=log.(xs[bs:be]); ly=log.(es[bs:be]); n=be-bs+1
                sl=-(n*sum(lx.*ly)-sum(lx)*sum(ly))/(n*sum(lx.^2)-sum(lx)^2)
                name*=@sprintf("  (slope %.1f)",sl)
            end
        end
        plot!(plt,xs,es;marker=mk,color=col,linestyle=ls,label=name,markersize=5,lw=2)
    end
    png(plt,ASSET)
    if isdir(PAPER)
        try; png(plt,joinpath(PAPER,"helical_milling_order.png")); savefig(plt,joinpath(PAPER,"helical_milling_order.pdf")); catch e; @warn e; end
    end
end

@printf("point Ω0=%.2f w=%.2f | refs: helical=%.6f smooth=%.6f hard(classical)=%.6f\n",
        Ω0,W,refs[:helical],refs[:smooth],refs[:hard]); flush(stdout)
for p in PS, m in MODES
    haskey(have,(m,p)) && continue
    ρ=ρcoll(m,p); have[(m,p)]=ρ; save(); render()
    @printf("  %-8s p=%4d  ρ=%.6f  err=%.2e\n",m,p,ρ,abs(ρ-refs[m])); flush(stdout)
end
render()
println("HELICAL MILLING ORDER DONE")
