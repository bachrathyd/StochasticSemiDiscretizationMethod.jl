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
        @test_logs (:warn, r"fractional-limit collocation engine") match_mode=:any spectralRadiusOfMapping_collocation(
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

        # (d) DELAYED MULTIPLICATIVE NOISE (β ≢ 0) — vT-full engine
        βsm(t) = @SMatrix [0.0 0.0; 0.12 0.0]      # delayed noise reads position (smooth)
        βrg(t) = @SMatrix [0.0 0.0; 0.0 0.12]      # delayed noise reads velocity (rough)
        mkβ(τ, βmx; B=Bpos) = LDDEProblem(ProportionalMX(Afun), [DelayMX(τ, B)],
            [stCoeffMX(1, ProportionalMX(αfun))], [stCoeffMX(1, DelayMX(τ, βmx))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        # aligned constant τ + β≢0: the function-valued vT-full path must reproduce
        # the aligned v8 engine (reached by the constant-τ path), smooth AND rough read
        for βmx in (βsm, βrg), S in (2, 3)
            ρv8 = spectralRadiusOfMapping_collocation(mkβ(0.5, βmx), T, 16; S=S)
            ρvT = spectralRadiusOfMapping_collocation(mkβ(t -> 0.5, βmx), T, 16; S=S,
                                                      verbosity=0)
            @test isapprox(ρv8, ρvT; rtol=1e-8)
        end
        # varying τ(t) + β≢0: measured order ≥ S+0.5 (S=2) for BOTH read types —
        # the delayed multiplicative noise carries no rough-read order collapse.
        # Delay τβ ≈ 0.3 (spans several steps) + a wide p-gap and an S=3 reference
        # keep the coarse-p errors well above the reference floor (a 2-point p=12/24
        # window against a p=64 S=3 ref sits ON the floor — negative slope noise).
        τβ(t) = 0.30 + 0.06sin(2π*t)
        for βmx in (βsm, βrg)
            ρref = spectralRadiusOfMapping_collocation(mkβ(τβ, βmx), T, 96; S=3,
                                                       verbosity=0)
            e_lo = abs(spectralRadiusOfMapping_collocation(mkβ(τβ, βmx), T, 12; S=2,
                                                           verbosity=0) - ρref)
            e_hi = abs(spectralRadiusOfMapping_collocation(mkβ(τβ, βmx), T, 48; S=2,
                                                           verbosity=0) - ρref)
            @test log(e_lo / e_hi) / log(4) ≥ 2.5
        end
        # β≢0 stationary variance through the unified API (no classical fallback)
        vβ = stationaryVariance(mkβ(τfun, βsm), T, 32; method=Collocation(3), verbosity=0)
        vβcl = stationaryVariance(mkβ(τfun, βsm), T, 800; method=ClassicalSD(2))
        @test isapprox(vβ, vβcl; rtol=3e-2)
        # block-boundary snap of a noise point-read (Finding 4): a constant delay
        # τ=(2+c₁)·Δt places stage 1's delayed image exactly on a block boundary,
        # so its noise read resolves to a neighbouring x_e rather than a D — a branch
        # the aligned reduction (interior-node reads) never reaches. Verify it runs
        # and self-converges to the fine-p value.
        c1 = StochasticSemiDiscretizationMethod.gl_tab(2)[3][1]
        τbnd = (2 + c1) / 16
        ρbnd = spectralRadiusOfMapping_collocation(mkβ(τbnd, βsm), T, 16; S=2, verbosity=0)
        ρbndref = spectralRadiusOfMapping_collocation(mkβ(τbnd, βsm), T, 64; S=2, verbosity=0)
        @test isfinite(ρbnd) && isapprox(ρbnd, ρbndref; rtol=1e-2)

        # (d2) GENUINE ERROR PATHS (still error regardless of β)
        @test_throws ErrorException spectralRadiusOfMapping_collocation(
            mkprob(τfun), T, 4; S=2, verbosity=0)           # τ(t) < Δt
        @test_throws ErrorException spectralRadiusOfMapping_collocation(
            mkprob(t -> 0.5 + 0.2sin(2π*t)), T, 16; S=2, verbosity=0)  # τ′ > 0.9
        # MULTIPLE INDEPENDENT WIENER CHANNELS — handled directly by the vT engine
        # (the per-channel α/β/σ sum independently in the Itô isometry, no fallback).
        # A physical channel split by 1/√2 into two INDEPENDENT copies must reproduce
        # the single-channel value bit-identically (variances add quadratically):
        #   Σ_c (α/√2)·(·)·(α/√2)ᵀ × 2 channels ≡ α·(·)·αᵀ.
        s2 = 1/sqrt(2)
        αh(t) = s2 .* αfun(t); βh(t) = s2 .* βsm(t)
        mkβ_split = LDDEProblem(ProportionalMX(Afun), [DelayMX(τfun, Bpos)],
            [stCoeffMX(1, ProportionalMX(αh)), stCoeffMX(2, ProportionalMX(αh))],
            [stCoeffMX(1, DelayMX(τfun, βh)), stCoeffMX(2, DelayMX(τfun, βh))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3*s2])),
                          stAdditive(2, Additive(@SVector [0.0, 0.3*s2]))])
        ρ_1ch = spectralRadiusOfMoment(mkβ(τfun, βsm), T, 48; method=Collocation(3), verbosity=0)
        ρ_2ch = spectralRadiusOfMoment(mkβ_split, T, 48; method=Collocation(3), verbosity=0)
        @test isapprox(ρ_1ch, ρ_2ch; rtol=1e-8)              # split ≡ single (exact operator)
        v_1ch = stationaryVariance(mkβ(τfun, βsm), T, 48; method=Collocation(3), verbosity=0)
        v_2ch = stationaryVariance(mkβ_split, T, 48; method=Collocation(3), verbosity=0)
        @test isapprox(v_1ch, v_2ch; rtol=1e-8)
        # distinct-role channels (ch1 multiplicative, ch2 additive) run and cross-check
        # loosely against the classical multi-channel path (physical-correctness sanity)
        βmc = LDDEProblem(ProportionalMX(Afun), [DelayMX(τfun, Bpos)],
            [stCoeffMX(1, ProportionalMX(αfun))], [stCoeffMX(1, DelayMX(τfun, βsm))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3])),
                          stAdditive(2, Additive(@SVector [0.0, 0.2]))])
        ρ_mc = spectralRadiusOfMoment(βmc, T, 200; method=Collocation(3), verbosity=0)
        @test isapprox(ρ_mc, spectralRadiusOfMoment(βmc, T, 200; method=ClassicalSD(2));
                       rtol=3e-2)

        # wrapper hygiene (reviewer round 2): (i) two present-noise terms on ONE
        # channel SUM — 2×[0.15] ≡ 1×[0.30] (both the constant `Prob` and the
        # function-valued ProbT paths); (ii) a β whose delay is absent from the
        # drift set ERRORS; (iii) a β at a τ shared by two B_j is counted ONCE.
        mkαf(s) = ProportionalMX(t -> @SMatrix [0.0 0.0; s 0.0])
        mk1(αts, τ) = LDDEProblem(ProportionalMX(Afun), [DelayMX(τ, Bpos)],
            αts, [stCoeffMX(1, DelayMX(τ, t -> z2))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        for τ in (0.5, τfun)     # Prob (constant) and ProbT (function) paths
            a15 = mkαf(0.15)
            ρ2 = spectralRadiusOfMoment(mk1([stCoeffMX(1, a15), stCoeffMX(1, a15)], τ),
                                        T, 32; method=Collocation(3), verbosity=0)
            ρ1 = spectralRadiusOfMoment(mk1([stCoeffMX(1, mkαf(0.30))], τ),
                                        T, 32; method=Collocation(3), verbosity=0)
            @test isapprox(ρ1, ρ2; rtol=1e-10)
        end
        badβ = LDDEProblem(ProportionalMX(Afun), [DelayMX(0.5, Bpos)],
            [stCoeffMX(1, ProportionalMX(αfun))],
            [stCoeffMX(1, DelayMX(0.7, βsm))],       # τ=0.7 ∉ drift delays {0.5}
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        @test_throws ErrorException spectralRadiusOfMoment(badβ, T, 32;
                                                           method=Collocation(3), verbosity=0)
        mkBf(s) = t -> @SMatrix [0.0 0.0; s 0.0]
        shared = LDDEProblem(ProportionalMX(Afun),
            [DelayMX(τfun, mkBf(0.12)), DelayMX(τfun, mkBf(0.08))],
            [stCoeffMX(1, ProportionalMX(αfun))], [stCoeffMX(1, DelayMX(τfun, βsm))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        merged = LDDEProblem(ProportionalMX(Afun), [DelayMX(τfun, mkBf(0.20))],
            [stCoeffMX(1, ProportionalMX(αfun))], [stCoeffMX(1, DelayMX(τfun, βsm))],
            Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])
        @test isapprox(spectralRadiusOfMoment(shared, T, 32; method=Collocation(3), verbosity=0),
                       spectralRadiusOfMoment(merged, T, 32; method=Collocation(3), verbosity=0);
                       rtol=1e-9)

        # (e) HARD-REGIME ANCHORS (reviewer round 1)
        # noise-off gate on a genuinely time-varying configuration: ρ(H) = ρ(U)²
        pbT = SSM.ProbT(2, 1.0, t -> 0.45 + 0.08sin(2π*t), 0.37, 0.53,
                        t -> Matrix(Afun(t)), t -> Matrix(Bpos(t)),
                        t -> zeros(2, 2), t -> zeros(2, 2))
        engT = SSM.build_vT(pbT, 2, 12; want_U=true)
        @test isapprox(SSM.rho_H_krylov_v9m(engT), SSM.rho_U_vT(engT)^2; rtol=1e-10)
        # delay longer than the period (r_buf > p: residue classes reused in-window)
        ρ9L = spectralRadiusOfMapping_collocation(mkprob(1.5), T, 8; S=2)
        ρTL = spectralRadiusOfMapping_collocation(mkprob(t -> 1.5), T, 8; S=2,
                                                  verbosity=0)
        @test isapprox(ρ9L, ρTL; rtol=1e-10)
        # fast-DECREASING delay (τ′ ≈ −1.1 ⇒ ξ′ up to ≈2.1, 3-block reading spans;
        # allowed — the τ′ ≤ 0.9 bound is one-sided)
        τfast(t) = 0.45 + 0.12sin(2π*t) - 0.03sin(4π*t)
        ρ_fast = spectralRadiusOfMapping_collocation(mkprob(τfast), T, 16; S=2,
                                                     verbosity=0)
        ρ_fcl = spectralRadiusOfMoment(mkprob(τfast), T, 400; method=ClassicalSD(2))
        @test isapprox(ρ_fast, ρ_fcl; rtol=1e-2)
    end

    @testset "multiple delays (vT g>1)" begin
        SSM = StochasticSemiDiscretizationMethod
        T = 1.0
        Am = t -> [0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]
        αm = t -> [0.0 0.0; 0.25 0.0]
        σm = t -> reshape([0.0, 0.3], 2, 1)
        zm = t -> zeros(2,2)
        smp(τf) = (s = τf.(range(0,T;length=129)); (minimum(s), maximum(s)))
        # direct ProbT (heterogeneous closures allowed; LDDEProblem needs uniform types)
        function mkPT(τfs, Bs, βs)
            mn=Float64[]; mx=Float64[]
            for τf in τfs; (a,b)=smp(τf); push!(mn,a); push!(mx,b); end
            SSM.ProbT(2, T, Function[τfs...], mn, mx, Am, Function[Bs...], αm,
                      Function[βs...], σm)
        end
        ρ(pt,S,p) = SSM.rho_H_krylov_v9m(SSM.build_vT(pt,S,p))

        # EXACT equal-delay reduction: {τ,B1,β1}+{τ,B2,β2} ≡ {τ,B1+B2,β1+β2}
        for τf in (t->0.5, t->0.30+0.05sin(2π*t)), S in (2,3)
            B1=t->[0.0 0.0; 0.15 0.0]; B2=t->[0.0 0.0; 0.07 0.0]
            β1=t->[0.0 0.0; 0.10 0.0]; β2=t->[0.0 0.0; 0.05 0.0]
            Bsum=t->[0.0 0.0; 0.22 0.0]; βsum=t->[0.0 0.0; 0.15 0.0]
            r2 = ρ(mkPT([τf,τf],[B1,B2],[β1,β2]), S, 16)
            r1 = ρ(mkPT([τf],[Bsum],[βsum]), S, 16)
            @test isapprox(r2, r1; rtol=1e-10)
        end

        # g=2 distinct varying delays, mixed β (delay 2 noise-free): high order
        let
            B1=t->[0.0 0.0; 0.18 0.0]; B2=t->[0.0 0.0; 0.09 0.0]
            β1=t->[0.0 0.0; 0.10 0.0]
            τ1=t->0.30+0.06sin(2π*t); τ2=t->0.50+0.05cos(2π*t)
            pt=mkPT([τ1,τ2],[B1,B2],[β1,zm])
            ρref=ρ(pt,3,64)
            e_lo=abs(ρ(pt,2,16)-ρref); e_hi=abs(ρ(pt,2,32)-ρref)
            @test log2(e_lo/e_hi) ≥ 2.5
        end

        # public API: constant 2-delay problem (uniform SMatrix) vs classical
        let
            prob = LDDEProblem(
                ProportionalMX(t -> @SMatrix [0.0 1.0; -1.0 -0.4]),
                [DelayMX(0.5,  @SMatrix [0.0 0.0; 0.15 0.0]),
                 DelayMX(0.75, @SMatrix [0.0 0.0; 0.08 0.0])],
                [stCoeffMX(1, ProportionalMX(@SMatrix [0.0 0.0; 0.2 0.0]))],
                [stCoeffMX(1, DelayMX(0.5,  @SMatrix [0.0 0.0; 0.1 0.0])),
                 stCoeffMX(1, DelayMX(0.75, @SMatrix [0.0 0.0; 0.05 0.0]))],
                Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))], 1)
            ρc = spectralRadiusOfMoment(prob, T, 24; method=Collocation(3), verbosity=0)
            ρs = spectralRadiusOfMoment(prob, T, 500; method=ClassicalSD(2))
            @test isapprox(ρc, ρs; rtol=1e-2)
            vc = stationaryVariance(prob, T, 24; method=Collocation(3), verbosity=0)
            vs = stationaryVariance(prob, T, 500; method=ClassicalSD(2))
            @test isapprox(vc, vs; rtol=2e-2)
        end
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
