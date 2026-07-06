using StochasticSemiDiscretizationMethod
using CUDA
using Test

function createHayesProblem(a,β)
    AMx =  ProportionalMX(a*ones(1,1));
    τ1=1. 
    BMx1 = DelayMX(τ1,zeros(1,1));
    cVec = Additive(1)
    noiseID = 1
    αMx1 = stCoeffMX(noiseID,ProportionalMX(zeros(1,1)))
    βMx11 = stCoeffMX(noiseID,DelayMX(τ1,β*ones(1,1)))
    σ = stAdditive(1,Additive(ones(1)))
    LDDEProblem(AMx,[BMx1],[αMx1],[βMx11],cVec,[σ])
end

function createSLDOProblem(A,B,ζ,α,β,σ)
    AMx =  ProportionalMX(@SMatrix [0. 1.;-A -2ζ]);
    τ1=2π 
    BMx1 = DelayMX(τ1,@SMatrix [0. 0.; B 0.]);
    cVec = Additive(2)
    noiseID = 1
    αMx1 = stCoeffMX(noiseID,ProportionalMX(@SMatrix [0. 0.; α 0.]))
    βMx11 = stCoeffMX(noiseID,DelayMX(τ1,@SMatrix [0. 0.; β 0.]))
    σVec = stAdditive(1,Additive(@SVector [0., σ]))
    LDDEProblem(AMx,[BMx1],[αMx1],[βMx11],cVec,[σVec])
end

function tests()
    @testset "Testing the SemiDiscretizationMethod package with the examples" begin
		#Hayes
        @test begin
            hayes_lddep=createHayesProblem(-6.,2.); # LDDE problem for Hayes equation
            method=SemiDiscretization(0,0.1) # 0th order semi discretization with Δt=0.1
            τmax=1. # the largest τ of the system
            # Second Moment mapping
            mapping=DiscreteMapping_M2(hayes_lddep,method,τmax,n_steps=10,calculate_additive=true); #The discrete mapping of the system

            spectralRadiusOfMapping(mapping); # spectral radius ρ of the mapping matrix (ρ>1 unstable, ρ<1 stable)
            statM2=VecToCovMx(fixPointOfMapping(mapping), length(mapping.M1_Vs[1])); # stationary second moment matrix of the hayes equation (equilibrium position)
            true
        end
		#Stochastic Delay Oscillator
        @test begin
            SLDOP_lddep=createSLDOProblem(1.,0.1,0.1,0.1,0.1,0.5); # LDDE problem for Hayes equation
            method=SemiDiscretization(5,(2π+100eps())/10) # 5th order semi discretization with Δt=2π/10
            τmax=2π # the largest τ of the system
            # Second Moment mapping
            mapping=DiscreteMapping_M2(SLDOP_lddep,method,τmax,n_steps=10,calculate_additive=true); #The discrete mapping of the system
            
            spectralRadiusOfMapping(mapping); # spectral radius ρ of the mapping matrix (ρ>1 unstable, ρ<1 stable)
            statM2=VecToCovMx(fixPointOfMapping(mapping), length(mapping.M1_Vs[1])); # stationary second moment matrix of the hayes equation (equilibrium position)
            true
        end
    end

    @testset "Multiplication-free (MF) path vs dense mapping" begin
        hayes_lddep = createHayesProblem(-6., 2.)
        method = SemiDiscretization(0, 0.1)
        mapping_dense = DiscreteMapping_M2(hayes_lddep, method, 1., n_steps=10)
        ρ_dense = spectralRadiusOfMapping(mapping_dense)

        rst = StochasticSemiDiscretizationMethod.calculateResults(
            hayes_lddep, method, 1., n_steps=10)
        dm_mf = DiscreteMapping_M2_MF(rst)
        ρ_mf = spectralRadiusOfMapping_MF(dm_mf)
        @test isapprox(ρ_mf, ρ_dense; rtol=1e-8)
    end

    @testset "Additive variance vs analytic (distinct Wiener channels)" begin
        # damped oscillator, B=0: stationary Var(x) = σ²/(4ζ) exactly.
        # The additive source lives on its OWN Wiener channel (nID=2), distinct
        # from the (zero) multiplicative channel (nID=1) — the case where a
        # source used to be duplicated into every channel slot (factor 2 in the
        # factored fixpoint, factor 4 in the classical one).
        ζ = 0.1; σ = 0.5; τ1 = 2π
        AMx = ProportionalMX(@SMatrix [0. 1.; -1. -2ζ])
        BMx = DelayMX(τ1, @SMatrix [0. 0.; 0. 0.])
        α0  = stCoeffMX(1, ProportionalMX(@SMatrix [0. 0.; 0. 0.]))
        β0  = stCoeffMX(1, DelayMX(τ1, @SMatrix [0. 0.; 0. 0.]))
        method = SemiDiscretization(2, (2π+100eps())/50)
        exact = σ^2/(4ζ)

        lddep = LDDEProblem(AMx, [BMx], [α0], [β0],
            Additive(2), [stAdditive(2, Additive(@SVector [0., σ]))])
        rst = StochasticSemiDiscretizationMethod.calculateResults(
            lddep, method, τ1, n_steps=50, calculate_additive=true)
        @test isapprox(fixPointOfMapping_MF_factored(rst)[1], exact; rtol=1e-2)
        @test isapprox(fixPointOfMapping(DiscreteMapping_M2(rst))[1], exact; rtol=1e-2)

        # two independent additive sources (used to throw DimensionMismatch):
        # variances add across channels.
        σ2 = 0.3
        lddep2 = LDDEProblem(AMx, [BMx], [α0], [β0],
            Additive(2), [stAdditive(2, Additive(@SVector [0., σ])),
                          stAdditive(3, Additive(@SVector [0., σ2]))])
        rst2 = StochasticSemiDiscretizationMethod.calculateResults(
            lddep2, method, τ1, n_steps=50, calculate_additive=true)
        @test isapprox(fixPointOfMapping_MF_factored(rst2)[1],
                       (σ^2+σ2^2)/(4ζ); rtol=1e-2)
    end

    # CUDA.functional() alone is not enough: the selected CUDA runtime may not
    # support the installed GPU (e.g. CUDA 13 dropped Pascal). Probe with a real
    # device round-trip before enabling the GPU tests. If the probe fails under
    # Pkg.test but the GPU works otherwise, pin the runtime in
    # test/LocalPreferences.toml ([CUDA_Runtime_jll] version = "...").
    gpu_usable = CUDA.functional() && try
        x = CUDA.zeros(Float64, 4)
        CUDA.@sync x .+= 1.0
        Array(x) == fill(1.0, 4)
    catch err
        @info "GPU present but unusable — skipping GPU tests" err
        false
    end

    @testset "GPU path vs CPU MF reference" begin
        if gpu_usable
            # scalar Hayes (d=1)
            hayes_lddep = createHayesProblem(-6., 2.)
            rst = StochasticSemiDiscretizationMethod.calculateResults(
                hayes_lddep, SemiDiscretization(0, 0.1), 1., n_steps=10)
            dm = DiscreteMapping_M2_MF(rst)
            ρ_cpu = spectralRadiusOfMapping_MF(dm)
            @test isapprox(spectralRadiusOfMapping_GPU(dm), ρ_cpu; rtol=1e-8)
            @test isapprox(spectralRadiusOfMapping_auto(dm), ρ_cpu; rtol=1e-8)

            # delay oscillator (d=2, multiplicative + delayed noise)
            SLDOP_lddep = createSLDOProblem(1., 0.1, 0.1, 0.1, 0.1, 0.5)
            rst2 = StochasticSemiDiscretizationMethod.calculateResults(
                SLDOP_lddep, SemiDiscretization(2, (2π+100eps())/20), 2π, n_steps=20)
            dm2 = DiscreteMapping_M2_MF(rst2)
            @test isapprox(spectralRadiusOfMapping_GPU(dm2),
                           spectralRadiusOfMapping_MF(dm2); rtol=1e-8)

            # delay-stochastic Mathieu with ALL matrices time-periodic
            # (A(t), B(t), α(t), β(t); P=4π, τ=2π) — the representative case:
            # every per-step coefficient tensor differs, so the whole GPU
            # coefficient pipeline is exercised.
            AMxfun(t) = @SMatrix [0. 1.; -(3.0 + 2.0*cos(0.5t)) -0.2]
            BMxfun(t) = @SMatrix [0. 0.; 0.5*(1+0.4cos(0.5t)) 0.]
            αMxfun(t) = @SMatrix [0. 0.; -0.1*(3.0 + 2.0*cos(0.5t)) -0.02]
            βMxfun(t) = @SMatrix [0. 0.; 0.05*(1+0.4cos(0.5t)) 0.]
            mathieu_lddep = LDDEProblem(
                ProportionalMX(AMxfun), [DelayMX(2π, BMxfun)],
                [stCoeffMX(1, ProportionalMX(αMxfun))],
                [stCoeffMX(1, DelayMX(2π, βMxfun))],
                Additive(2), [stAdditive(1, Additive(@SVector [0., 0.]))])
            rst4 = StochasticSemiDiscretizationMethod.calculateResults(
                mathieu_lddep, SemiDiscretization(2, 4π/24), 2π, n_steps=24)
            dm4 = DiscreteMapping_M2_MF(rst4)
            @test isapprox(spectralRadiusOfMapping_GPU(dm4),
                           spectralRadiusOfMapping_MF(dm4); rtol=1e-8)

            # stationary second moment (additive noise)
            rst3 = StochasticSemiDiscretizationMethod.calculateResults(
                hayes_lddep, SemiDiscretization(0, 0.1), 1.,
                n_steps=10, calculate_additive=true)
            dm3 = DiscreteMapping_M2_MF(rst3)
            m_cpu = fixPointOfMapping_MF(dm3)
            m_gpu = Array(fixPointOfMapping_GPU(dm3))
            @test maximum(abs.(m_cpu .- m_gpu)) < 1e-8 * maximum(abs.(m_cpu))
        else
            @info "CUDA not usable — skipping GPU tests"
            @test true
        end
    end
end
tests()
