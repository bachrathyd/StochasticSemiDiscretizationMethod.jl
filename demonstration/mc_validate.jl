# Validate the Monte-Carlo estimator on the SCALAR HAYES equation, where the SDM value
# is trustworthy (q=2 → 0.1486) and the 2nd moment obeys a clean structure.
# If MC matches Hayes, the MC is reliable and we can trust it on Mathieu; if not, the
# MC estimator itself is buggy (then SDM stays our best reference).
using Random, Statistics, Printf, LinearAlgebra

# Hayes: dx = (A x + B x(t-1)) dt + (β x(t-1)) dW,  A=-1,B=-0.4,β=0.3, τ=1, "period" P=1.
const A=-1.0; const B=-0.4; const BETA=0.3; const TAUh=1.0; const Ph=1.0
const SDM_HAYES = 0.1486262   # trusted SDM q=4 reference (scalar, validated)

# scalar EM: x_{n+1} = x_n + dt(A x_n + B xτ) + β xτ dW
function mc_hayes(; Nper=60, nsub=2000, npath=20000, seed=7, burn=15)
    rng=MersenneTwister(seed)
    dt=Ph/nsub; rstep=round(Int,TAUh/dt); nsteps=Nper*nsub; sdt=sqrt(dt)
    ms=zeros(Nper+1)
    for ip in 1:npath
        buf=Vector{Float64}(undef,nsteps+rstep+1)
        for k in 1:rstep+1; buf[k]=1.0; end
        ms[1]+=1.0; cur=rstep+1; pidx=1
        for n in 1:nsteps
            xτ=buf[cur-rstep]; x=buf[cur]; dW=sdt*randn(rng)
            xn = x + dt*(A*x+B*xτ) + BETA*xτ*dW
            cur+=1; buf[cur]=xn
            if n%nsub==0; pidx+=1; ms[pidx]+=xn^2; end
        end
    end
    ms./=npath
    ratios=[ms[k+1]/ms[k] for k in (burn+1):Nper]
    exp(mean(log.(ratios)))
end

println("MC validation on scalar HAYES (trusted SDM value = $SDM_HAYES):")
for (np,ns,nsub) in [(8000,60,1000),(20000,60,2000),(40000,80,3000)]
    ρ=mc_hayes(npath=np,Nper=ns,nsub=nsub)
    @printf("  npath=%5d Nper=%2d nsub=%4d  →  ρ_MC=%.5f   (Δ vs SDM = %+.4f)\n",
            np,ns,nsub,ρ,ρ-SDM_HAYES)
end
println("\nIf ρ_MC → 0.1486, the MC estimator is correct ⇒ trust it on Mathieu.")
println("If ρ_MC stays far from 0.1486, the MC estimator is biased ⇒ SDM remains the reference.")
