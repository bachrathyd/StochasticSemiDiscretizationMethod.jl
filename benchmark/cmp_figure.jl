# Combined comparison figure vs Iklodi & Dankowicz (arXiv:2607.01374):
#   (a) accuracy: GL-S collocation error against their EXACT scalar χ=0 boundary
#   (b) CPU: our MF-SSDM second-moment turning lobe (their Fig. 13), 75 s / 1 core
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod, StaticArrays, Printf, Plots, DelimitedFiles
BLAS.set_num_threads(1)

# ---- (a) scalar accuracy vs exact boundary χ=0 (τ=1, α=-1.5, β=0) ----
chi(a,b;τ=1.0,α=-1.5,β=0.0)=begin μ=(a^2-b^2)≥0 ? sqrt(a^2-b^2) : sqrt(complex(a^2-b^2))
    real(((α+β)^2+2a+2b)*cosh(μ*τ/2)-((a+b)/μ)*((α-β)^2+2a-2b)*sinh(μ*τ/2)) end
function bstar(a;lo=1.0,hi=2.99); flo=chi(a,lo)
    for _ in 1:80; m=(lo+hi)/2; (flo*chi(a,m)≤0) ? (hi=m) : (lo=m;flo=chi(a,m)); end; (lo+hi)/2 end
sp(a,b;τ=1.0,α=-1.5)=LDDEProblem(ProportionalMX(t->@SMatrix [a]),[DelayMX(τ,t->@SMatrix [b])],
    [stCoeffMX(1,ProportionalMX(t->@SMatrix [α]))],[stCoeffMX(1,DelayMX(τ,t->@SMatrix [0.0]))],
    Additive(1),[stAdditive(1,Additive(@SVector [0.0]))])
a0=-3.0; b0=bstar(a0); prob=sp(a0,b0)
scol=Dict(1=>:dodgerblue,2=>:seagreen,3=>:purple,4=>:darkorange,5=>:crimson)
PS=[2,3,4,5,6,8,10,12,16,20,24,32,48,64,96,128]
pa=plot(xscale=:log10,yscale=:log10,framestyle=:box,xlabel="p  (steps per delay)",
        ylabel="|ρ(H) − 1|   vs exact χ=0 boundary",title="(a)  accuracy vs the exact analytic boundary",
        titlefontsize=10,legend=:bottomleft,legendfontsize=7,guidefontsize=9,tickfontsize=8)
for (ord,anch) in ((2,3e-1),(4,3e-2)); pg=[2.0,128.0]; plot!(pa,pg,anch.*(2.0./pg).^ord,color=:gray80,ls=:dot,lw=1,label= ord==2 ? "O(p⁻²)/O(p⁻⁴)" : "")
end
hline!(pa,[2e-11],color=:gray,ls=:dashdot,lw=1,label="solver floor")
for S in 1:5
    ps=Int[]; er=Float64[]
    for p in PS
        e=abs(spectralRadiusOfMapping_collocation(prob,1.0,p;S=S)-1); push!(ps,p); push!(er,max(e,1e-16))
        count(<(2e-12),er)≥3 && break
    end
    plot!(pa,ps,er,marker=:utriangle,ms=3.5,lw=1.5,color=scol[S],label="GL$S")
end

# ---- (b) turning second-moment lobe (our boundary + their labelled points) ----
raw,_=readdlm(joinpath(@__DIR__,"cmp_turning_lobe.csv"),',';header=true)
pb=scatter(raw[:,1],raw[:,2],ms=1.6,color=:crimson,markerstrokewidth=0,label="MF-SSDM  ρ(H)=1",
        framestyle=:box,xlabel="ω̃  (dimensionless spindle speed)",ylabel="w̃  (chip width)",
        title="(b)  second-moment lobe — 774 pts in 75 s (1 core)",titlefontsize=10,
        legend=:topleft,legendfontsize=7,guidefontsize=9,tickfontsize=8,xlims=(0.1,2.0),ylims=(0,0.8))
scatter!(pb,[1.0,1.0,0.60],[0.10,0.55,0.20],marker=:xcross,ms=6,color=:black,label="A,B,C (ref.)")
annotate!(pb,[(1.02,0.12,text("A",8,:left)),(1.02,0.57,text("B",8,:left)),(0.62,0.22,text("C",8,:left))])
plt=plot(pa,pb,layout=(1,2),size=(1080,430),dpi=200,left_margin=5Plots.mm,bottom_margin=5Plots.mm)
savefig(plt,joinpath(@__DIR__,"cmp_iklodi.png")); savefig(plt,joinpath(@__DIR__,"cmp_iklodi.pdf"))
println("done — cmp_iklodi.png")
