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

    @testset "High-order Gauss–Legendre collocation (order 2S)" begin
        # delayed-PD-drift stochastic Mathieu (β ≡ 0 ⇒ pruned engine), additive noise
        Afun(t) = @SMatrix [0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]
        Bfun(t) = @SMatrix [0.0 0.0; 0.20*(1+0.3cos(2π*t)) 0.12*(1+0.4cos(2π*t))]
        αfun(t) = @SMatrix [0.0 0.0; 0.30 0.0]
        βfun(t) = @SMatrix [0.0 0.0; 0.0 0.0]
        prob = LDDEProblem(ProportionalMX(Afun), [DelayMX(1.0, Bfun)],
            [stCoeffMX(1, ProportionalMX(αfun))], [stCoeffMX(1, DelayMX(1.0, βfun))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        T = 1.0
        ρref = spectralRadiusOfMapping_collocation(prob, T, 64; S=3)
        @test 0.0 < ρref < 1.0
        # measured convergence order ≈ 2S for S = 1, 2, 3.
        # Use coarse resolutions p = 4, 8: at S = 3 (order 6) the error already
        # bottoms out at the ρref floor (~1e-12) by p ≈ 12, so a finer p_hi makes
        # the Richardson order estimate meaningless floating-point noise. At p = 8
        # the S = 3 error is still ~1e-9 (≈100× above the floor), giving a genuine
        # order measurement (S=1,2,3 → 1.99, 4.00, 6.03).
        for S in (1, 2, 3)
            e_lo = abs(spectralRadiusOfMapping_collocation(prob, T, 4; S=S) - ρref)
            e_hi = abs(spectralRadiusOfMapping_collocation(prob, T, 8; S=S) - ρref)
            order = log(e_lo / e_hi) / log(8 / 4)
            @test isapprox(order, 2S; atol=0.4)
        end
        # stationary variance is finite and positive
        Var = fixPointOfMapping_collocation(prob, T, 32; S=3)[1, 1]
        @test isfinite(Var) && Var > 0
    end

    @testset "collocation precomputed noise operator == reference (v9)" begin
        # The per-step precomputed noise operator (the fast Krylov path) must
        # reproduce the reference implementation that rebuilds noise_block every
        # matvec, to solver tolerance — on a random symmetric covariance.
        SSM = StochasticSemiDiscretizationMethod
        Afun(t) = @SMatrix [0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]
        Bfun(t) = @SMatrix [0.0 0.0; 0.20*(1+0.3cos(2π*t)) 0.0]
        αfun(t) = @SMatrix [0.0 0.0; 0.30 0.0]
        βfun(t) = @SMatrix [0.0 0.0; 0.0 0.0]           # β ≡ 0 ⇒ pruned v9 engine
        prob = LDDEProblem(ProportionalMX(Afun), [DelayMX(1.0, Bfun)],
            [stCoeffMX(1, ProportionalMX(αfun))], [stCoeffMX(1, DelayMX(1.0, βfun))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        eng = SSM.build_v9m(SSM._collocation_prob(prob, 1.0), 3, 16)
        W = eng.W
        Rm = randn(W, W); C = (Rm + Rm') / 2
        slow = SSM._applyH_v9m_slow(eng, C)             # rebuild-every-call reference
        fast = SSM.applyH_v9m(eng, C)                   # precomputed-ops path
        scale = max(1.0, maximum(abs, slow))
        @test maximum(abs, fast - slow) < 1e-9 * scale
        ws = SSM.V9Workspace(eng)                       # ring-buffer workspace path
        copyto!(ws.C, C)
        res = SSM._applyH_period!(ws, eng, ws.C)
        @test maximum(abs, res - slow) < 1e-9 * scale
    end

    @testset "Unified method selection (collocation vs classical SD)" begin
        Afun(t) = @SMatrix [0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]
        Bfun(t) = @SMatrix [0.0 0.0; 0.20*(1+0.3cos(2π*t)) 0.12*(1+0.4cos(2π*t))]
        αfun(t) = @SMatrix [0.0 0.0; 0.30 0.0]
        βfun(t) = @SMatrix [0.0 0.0; 0.0 0.0]
        prob = LDDEProblem(ProportionalMX(Afun), [DelayMX(1.0, Bfun)],
            [stCoeffMX(1, ProportionalMX(αfun))], [stCoeffMX(1, DelayMX(1.0, βfun))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        T = 1.0
        # default method is Collocation(3)
        @test spectralRadiusOfMoment(prob, T, 12) ==
              spectralRadiusOfMoment(prob, T, 12; method=Collocation(3))
        @test GaussLegendre(3) === Collocation(3)
        # collocation (few steps) and classical SD (many steps) agree on ρ and Var
        ρ_gl = spectralRadiusOfMoment(prob, T, 16; method=GaussLegendre(3))
        ρ_sd = spectralRadiusOfMoment(prob, T, 400; method=ClassicalSD(2))
        @test isapprox(ρ_gl, ρ_sd; rtol=2e-2)
        v_gl = stationaryVariance(prob, T, 16; method=Collocation(3))
        v_sd = stationaryVariance(prob, T, 400; method=ClassicalSD(2))
        @test isapprox(v_gl, v_sd; rtol=3e-2)
    end

    @testset "T/τ ratios (τ<T, τ=T, τ>T) — collocation vs classical SD" begin
        make(τ) = LDDEProblem(
            ProportionalMX(t -> @SMatrix [0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]),
            [DelayMX(τ, t -> @SMatrix [0.0 0.0; 0.15 0.08])],
            [stCoeffMX(1, ProportionalMX(t -> @SMatrix [0.0 0.0; 0.25 0.0]))],
            [stCoeffMX(1, DelayMX(τ, t -> @SMatrix zeros(2,2)))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        T = 1.0
        for (ratio, p) in ((0.5, 24), (1.0, 16), (2.0, 16))
            prob = make(ratio*T)
            @test isapprox(ratio*p, round(ratio*p); atol=1e-9)   # r = τ·p/T integer
            ρ_gl = spectralRadiusOfMoment(prob, T, p;   method=GaussLegendre(3))
            ρ_sd = spectralRadiusOfMoment(prob, T, 500; method=ClassicalSD(2))
            @test isapprox(ρ_gl, ρ_sd; rtol=2e-2)
        end
    end

    @testset "time-varying delay collocation (vT engine)" begin
        SSM = StochasticSemiDiscretizationMethod
        # Mathieu-type smooth test problem; `mkB` selects the delayed-read
        # structure: :pos reads the (smooth) position, :vel the (Wiener-rough)
        # velocity — the preintegrated history must handle both at full order.
        Afun(t) = @SMatrix [0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]
        Bpos(t) = @SMatrix [0.0 0.0; 0.20*(1+0.3cos(2π*t)) 0.0]
        Bvel(t) = @SMatrix [0.0 0.0; 0.0 0.18*(1+0.3cos(2π*t))]
        αfun(t) = @SMatrix [0.0 0.0; 0.25 0.0]
        z2 = @SMatrix zeros(2,2)
        mkprob(τ; B=Bpos) = LDDEProblem(ProportionalMX(Afun), [DelayMX(τ, B)],
            [stCoeffMX(1, ProportionalMX(αfun))], [stCoeffMX(1, DelayMX(τ, t -> z2))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        T = 1.0

        # (a) ALIGNED REDUCTION: a constant delay passed as a *function* routes
        # through the vT engine and must reproduce the aligned v9 engine.
        for S in (1, 2, 3)
            ρ9 = spectralRadiusOfMapping_collocation(mkprob(0.5), T, 16; S=S)
            ρT = spectralRadiusOfMapping_collocation(mkprob(t -> 0.5), T, 16; S=S,
                                                     verbosity=0)
            @test isapprox(ρ9, ρT; rtol=1e-10)
        end
        v9 = fixPointOfMapping_collocation(mkprob(0.5), T, 16; S=3)[1, 1]
        vT = fixPointOfMapping_collocation(mkprob(t -> 0.5), T, 16; S=3, verbosity=0)[1, 1]
        @test isapprox(v9, vT; rtol=1e-9)

        # warning contract (first verbosity=1 uses in this session: the sites
        # below fire exactly once thanks to maxlog=1 — test them here, in order)
        @test_logs (:warn, r"function-valued delay") match_mode=:any spectralRadiusOfMapping_collocation(
            mkprob(t -> 0.5), T, 8; S=1, verbosity=1)
        @test_logs (:warn, r"not an integer multiple") match_mode=:any spectralRadiusOfMapping_collocation(
            mkprob(0.5 + 1e-3), T, 8; S=1, verbosity=1)

        # (b) CONSTANT INCOMMENSURATE delay (golden ratio — never grid-aligned):
        # converges to the same limit as the classical path, error shrinking
        # against a finer vT self-reference.
        τg = (sqrt(5.0) - 1.0) / 2.0
        ρ_ref_g = spectralRadiusOfMapping_collocation(mkprob(τg), T, 48; S=3, verbosity=0)
        ρ_cl_g  = spectralRadiusOfMoment(mkprob(τg), T, 800; method=ClassicalSD(2))
        @test isapprox(ρ_ref_g, ρ_cl_g; rtol=2e-2)
        e12 = abs(spectralRadiusOfMapping_collocation(mkprob(τg), T, 12; S=2, verbosity=0) - ρ_ref_g)
        e24 = abs(spectralRadiusOfMapping_collocation(mkprob(τg), T, 24; S=2, verbosity=0) - ρ_ref_g)
        @test e24 < e12

        # (c) SMOOTH TIME-VARYING τ(t): measured order ≥ S+1 floor (with slack
        # 0.5) for both smooth (position) and rough (velocity) delayed reads —
        # the fractional-limit preintegrated history removes the rough-read cap.
        τfun(t) = 0.2 + 0.05sin(2π*t)
        for Bsel in (Bpos, Bvel)
            probv(τ) = mkprob(τ; B=Bsel)
            ρref = spectralRadiusOfMapping_collocation(probv(τfun), T, 64; S=3, verbosity=0)
            for (S, plo, phi, floor_order) in ((1, 16, 32, 1.5), (2, 8, 32, 2.5))
                e_lo = abs(spectralRadiusOfMapping_collocation(probv(τfun), T, plo; S=S,
                                                               verbosity=0) - ρref)
                e_hi = abs(spectralRadiusOfMapping_collocation(probv(τfun), T, phi; S=S,
                                                               verbosity=0) - ρref)
                order = log(e_lo / e_hi) / log(phi / plo)
                @test order ≥ floor_order
            end
        end
        # variance through the unified API on the varying-delay problem
        vvar = stationaryVariance(mkprob(τfun), T, 32; method=Collocation(3), verbosity=0)
        vcl  = stationaryVariance(mkprob(τfun), T, 800; method=ClassicalSD(2))
        @test isapprox(vvar, vcl; rtol=2e-2)

        # (d) ERROR PATHS + CLASS-6 FALLBACK
        # β ≢ 0 with a varying delay: the direct wrapper refuses …
        βprob = LDDEProblem(ProportionalMX(Afun), [DelayMX(τfun, Bpos)],
            [stCoeffMX(1, ProportionalMX(αfun))],
            [stCoeffMX(1, DelayMX(τfun, t -> @SMatrix [0.0 0.0; 0.1 0.0]))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        @test_throws ErrorException spectralRadiusOfMapping_collocation(
            βprob, T, 16; S=2, verbosity=0)
        # … while the unified API falls back to the classical path (identical
        # result to requesting ClassicalSD(2) explicitly)
        ρ_fb = spectralRadiusOfMoment(βprob, T, 200; method=Collocation(3), verbosity=0)
        ρ_cd = spectralRadiusOfMoment(βprob, T, 200; method=ClassicalSD(2))
        # same code path, but the Krylov start vector is random ⇒ solver-tol agreement
        @test isapprox(ρ_fb, ρ_cd; rtol=1e-9)
        # τ(t) < Δt: n_steps below the admissible minimum
        @test_throws ErrorException spectralRadiusOfMapping_collocation(
            mkprob(τfun), T, 4; S=2, verbosity=0)
        # non-monotone reading map (|τ′| too large)
        @test_throws ErrorException spectralRadiusOfMapping_collocation(
            mkprob(t -> 0.3 + 0.25sin(2π*t)), T, 16; S=2, verbosity=0)
    end

    @testset "time-periodic (cyclostationary) variance profile" begin
        # τ = 0.5 < T = 1 ⇒ buffer padded to a full period internally
        prob = LDDEProblem(
            ProportionalMX(t -> @SMatrix [0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]),
            [DelayMX(0.5, t -> @SMatrix [0.0 0.0; 0.15 0.08])],
            [stCoeffMX(1, ProportionalMX(t -> @SMatrix [0.0 0.0; 0.25 0.0]))],
            [stCoeffMX(1, DelayMX(0.5, t -> @SMatrix zeros(2,2)))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        T = 1.0; p = 40
        t, v = timePeriodicVariance(prob, T, p)
        @test length(t) == p && length(v) == p
        @test all(x -> isfinite(x) && x > 0, v)
        # phase-0 equals the single-phase stationary variance
        @test isapprox(v[1], stationaryVariance(prob, T, p; method=ClassicalSD(2)); rtol=1e-8)
        # genuinely varies over the period
        @test (maximum(v) - minimum(v)) / maximum(v) > 1e-3
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
