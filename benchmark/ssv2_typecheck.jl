# Verify the milling-matrix construction is type stable (no typed_hvcat / Any /
# Union), comparing the OLD block-hvcat form against the NEW column-major form.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StaticArrays, LinearAlgebra, InteractiveUtils
const N_TEETH=2; const PHI_EX=π/4; const Kr=0.3; const ζ=0.02
const Z2 = @SMatrix zeros(2,2); const I2 = SMatrix{2,2}(1.0I)

function Hdir(t,Ω0,Tssv,rva)
    φ0 = rva==0 ? Ω0*t : Ω0*t-(Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv)-1.0)
    h11=0.0;h12=0.0;h21=0.0;h22=0.0
    for j in 0:N_TEETH-1
        φ=mod(φ0+2π*j/N_TEETH,2π); φ ≤ PHI_EX || continue
        s,c=sincos(φ); a1=(c+Kr*s); a2=(s-Kr*c)
        h11+=a1*s;h12+=a1*c;h21+=-a2*s;h22+=-a2*c
    end
    @SMatrix [h11 h12; h21 h22]
end

Af_old(t,Ω0,Tssv,rva,w) = SMatrix{4,4}([Z2 I2; (-I2 .- w.*Hdir(t,Ω0,Tssv,rva)) (-2ζ).*I2])
Af_new(t,Ω0,Tssv,rva,w) = (H=Hdir(t,Ω0,Tssv,rva); SMatrix{4,4,Float64}(
    0.0,0.0,-1-w*H[1,1],-w*H[2,1],  0.0,0.0,-w*H[1,2],-1-w*H[2,2],
    1.0,0.0,-2ζ,0.0,  0.0,1.0,0.0,-2ζ))

# equivalence check
a = Af_old(0.37, 1.0, 62.8, 0.25, 0.7); b = Af_new(0.37, 1.0, 62.8, 0.25, 0.7)
println("max |Af_new - Af_old| = ", maximum(abs, a .- b))

sig = Tuple{Float64,Float64,Float64,Float64,Float64}
for (nm,fn) in (("OLD (block hvcat)",Af_old), ("NEW (column-major)",Af_new))
    io = IOBuffer(); code_warntype(io, fn, sig); s = String(take!(io))
    body = match(r"Body::[^\n]*", s)
    println("\n== $nm ==")
    println("  ", body === nothing ? "?" : body.match)
    println("  contains ::Any   : ", occursin("::Any", s))
    println("  contains Union   : ", occursin("Union", s))
    println("  contains hvcat   : ", occursin("hvcat", s))
end

# allocation comparison (per single call)
using BenchmarkTools
println("\nallocations per call:")
println("  OLD: ", (@allocated Af_old(0.37,1.0,62.8,0.25,0.7)), " bytes")
println("  NEW: ", (@allocated Af_new(0.37,1.0,62.8,0.25,0.7)), " bytes")
println("done")
