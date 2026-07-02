# =============================================================================
# Catalog of 20+ stochastic delay differential equation examples spanning
# engineering, biology, physics, control, economics & ecology — exercising:
#   • additive AND multiplicative noise
#   • T > τ  and  T < τ  (period vs delay)
#   • single, multiple, and TIME-VARYING delays
#   • constant and periodic (time-varying) coefficients
#   • higher-dimensional models (d = 1..4)
#   • various excitation forms
#
# Each entry returns (name, category, SDDEProblem, notes). The work-precision driver
# computes ρ(H) for GL(1..3) (moment-collocation, high order) and reports convergence.
#
# requires moment_engine.jl + moment_engine2.jl loaded first.
# =============================================================================

# convenience builders
sm1(x) = reshape([Float64(x)],1,1)
const I2 = [1.0 0.0;0.0 1.0]

# Each example: a NamedTuple (name, cat, prob, notes)
function build_examples()
    ex = NamedTuple[]

    # ---------- 1. Scalar stochastic Hayes (delayed multiplicative noise), T=τ ----------
    push!(ex, (name="01_hayes_scalar", cat="Math/benchmark",
        prob=SDDEProblem(1, 1.0, t->sm1(-1.0), [(t->1.0, t->sm1(-0.4))],
            [(t->sm1(0.0), [t->sm1(0.3)], t->[0.0])]),
        notes="Scalar Hayes dx=(Ax+Bx(t-1))dt+βx(t-1)dW. Analytic benchmark. T=τ."))

    # ---------- 2. Scalar Hayes with present + delayed multiplicative noise ----------
    push!(ex, (name="02_hayes_present_delay_noise", cat="Math/benchmark",
        prob=SDDEProblem(1, 1.0, t->sm1(-1.0), [(t->1.0, t->sm1(-0.4))],
            [(t->sm1(0.3), [t->sm1(0.2)], t->[0.0])]),
        notes="Hayes with both present (α) and delayed (β) multiplicative noise. T=τ."))

    # ---------- 3. Scalar Hayes additive noise ----------
    push!(ex, (name="03_hayes_additive", cat="Math/benchmark",
        prob=SDDEProblem(1, 1.0, t->sm1(-1.2), [(t->1.0, t->sm1(-0.5))],
            [(t->sm1(0.0), [t->sm1(0.0)], t->[0.4])]),
        notes="Hayes with purely additive noise σdW. Spectral radius governed by drift."))

    # ---------- 4. Scalar delayed logistic (population biology), multiplicative ----------
    # Linearized delayed logistic: dx=(-a x(t-τ))dt + α x dW
    push!(ex, (name="04_delayed_logistic", cat="Biology",
        prob=SDDEProblem(1, 1.0, t->sm1(0.0), [(t->1.0, t->sm1(-1.3))],
            [(t->sm1(0.15), [t->sm1(0.0)], t->[0.0])]),
        notes="Linearized delayed logistic (Hutchinson) ẋ=-a x(t-τ)+noise; population dynamics."))

    # ---------- 5. Stochastic delayed Mathieu (engineering, parametric), T>τ ----------
    push!(ex, (name="05_mathieu_TgtTau", cat="Engineering/parametric",
        prob=SDDEProblem(2, 4π,
            t->[0.0 1.0; -(3.0+2.0*cos(0.5*t)) -0.2],
            [(t->2π, t->[0.0 0.0; 0.5 0.0])],
            [(t->[0.0 0.0; -0.1*(3.0+2.0*cos(0.5*t)) -0.1*0.2],
              [t->[0.0 0.0; 0.1*0.5 0.0]], t->[0.0,0.0])]),
        notes="Stochastic delayed Mathieu, period P=4π > delay τ=2π. Multiplicative noise."))

    # ---------- 6. Stochastic delayed Mathieu with T<τ ----------
    push!(ex, (name="06_mathieu_TltTau", cat="Engineering/parametric",
        prob=SDDEProblem(2, 2π,
            t->[0.0 1.0; -(3.0+2.0*cos(t)) -0.2],
            [(t->4π, t->[0.0 0.0; 0.4 0.0])],            # τ=4π > period 2π
            [(t->[0.0 0.0; -0.1*(3.0+2.0*cos(t)) 0.0],
              [t->[0.0 0.0; 0.04 0.0]], t->[0.0,0.0])]),
        notes="Mathieu variant with delay τ=4π > period 2π (T<τ). Tests long-delay window."))

    # ---------- 7. Damped delayed oscillator, additive noise ----------
    push!(ex, (name="07_delayed_oscillator_additive", cat="Engineering/vibration",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; -4.0 -0.3],
            [(t->2π, t->[0.0 0.0; -0.5 0.0])],
            [(t->zeros(2,2), [t->zeros(2,2)], t->[0.0,0.3])]),
        notes="Delayed oscillator ẍ+0.3ẋ+4x=-0.5x(t-τ)+additive noise. Vibration model."))

    # ---------- 8. Delayed oscillator, multiplicative (velocity) noise ----------
    push!(ex, (name="08_delayed_oscillator_mult", cat="Engineering/vibration",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; -4.0 -0.3],
            [(t->2π, t->[0.0 0.0; -0.5 0.0])],
            [(t->[0.0 0.0; 0.0 0.2], [t->zeros(2,2)], t->[0.0,0.0])]),
        notes="Delayed oscillator with multiplicative velocity noise (0.2·ẋ·dW)."))

    # ---------- 9. Turning / regenerative machining chatter (engineering) ----------
    # ẍ + 2ζω ẋ + ω² x = -w (x - x(t-τ)) + noise (regenerative cutting)
    push!(ex, (name="09_machining_chatter", cat="Engineering/manufacturing",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; -(1.0+0.3) -2*0.05*1.0],
            [(t->2π, t->[0.0 0.0; 0.3 0.0])],
            [(t->[0.0 0.0; 0.05 0.0], [t->[0.0 0.0; -0.05 0.0]], t->[0.0,0.0])]),
        notes="Regenerative machine-tool chatter (turning). Surface-regeneration delay + noise."))

    # ---------- 10. Two-delay scalar system ----------
    push!(ex, (name="10_two_delay_scalar", cat="Math/multi-delay",
        prob=SDDEProblem(1, 1.0, t->sm1(-0.5),
            [(t->0.5, t->sm1(-0.3)), (t->1.0, t->sm1(-0.2))],
            [(t->sm1(0.1), [t->sm1(0.05), t->sm1(0.05)], t->[0.0])]),
        notes="Scalar with TWO delays τ₁=0.5, τ₂=1.0 and noise on both. Multi-delay."))

    # ---------- 11. Time-varying delay scalar ----------
    push!(ex, (name="11_time_varying_delay", cat="Math/time-varying-delay",
        prob=SDDEProblem(1, 2π, t->sm1(-0.8),
            [(t->1.0+0.3*sin(t), t->sm1(-0.4))],
            [(t->sm1(0.15), [t->sm1(0.1)], t->[0.0])]),
        notes="Scalar with periodic TIME-VARYING delay τ(t)=1+0.3 sin t. Multiplicative noise."))

    # ---------- 12. Neural field / delayed feedback (neuroscience) ----------
    push!(ex, (name="12_neural_delayed_feedback", cat="Biology/neuroscience",
        prob=SDDEProblem(1, 1.0, t->sm1(-1.0), [(t->0.8, t->sm1(0.6))],
            [(t->sm1(0.0), [t->sm1(0.2)], t->[0.1])]),
        notes="Delayed neural feedback (Mackey-Glass-type linearization) with noise."))

    # ---------- 13. Gene regulatory network (biology), delayed repression ----------
    push!(ex, (name="13_gene_regulatory", cat="Biology/systems",
        prob=SDDEProblem(2, 2π, t->[-0.5 0.0; 1.0 -0.5],
            [(t->1.5, t->[0.0 -0.8; 0.0 0.0])],
            [(t->[0.1 0.0;0.0 0.1], [t->zeros(2,2)], t->[0.05,0.05])]),
        notes="2-gene regulatory network with delayed repression + intrinsic noise. T<τ? T=2π>1.5."))

    # ---------- 14. Population predator-prey with delay (ecology) ----------
    push!(ex, (name="14_predator_prey_delay", cat="Ecology",
        prob=SDDEProblem(2, 2π, t->[0.1 -0.4; 0.3 -0.1],
            [(t->1.0, t->[-0.2 0.0; 0.0 0.0])],
            [(t->[0.1 0.0;0.0 0.05], [t->zeros(2,2)], t->[0.0,0.0])]),
        notes="Lotka–Volterra-type predator-prey, gestation delay, demographic noise."))

    # ---------- 15. Traffic flow car-following with reaction delay (engineering) ----------
    push!(ex, (name="15_traffic_carfollowing", cat="Engineering/traffic",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; 0.0 -0.6],
            [(t->1.2, t->[0.0 0.0; -0.9 -0.4])],
            [(t->[0.0 0.0;0.0 0.15], [t->zeros(2,2)], t->[0.0,0.1])]),
        notes="Car-following model with driver reaction delay τ=1.2. Speed-noise + additive."))

    # ---------- 16. Delayed feedback control loop (control engineering) ----------
    push!(ex, (name="16_feedback_control", cat="Engineering/control",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; -1.0 0.0],
            [(t->0.5, t->[0.0 0.0; -2.0 -0.5])],
            [(t->[0.0 0.0; 0.1 0.0], [t->[0.0 0.0; -0.1 0.0]], t->[0.0,0.05])]),
        notes="PD feedback control with actuation delay τ=0.5. Sensor + actuation noise."))

    # ---------- 17. Stochastic delayed Mathieu, strong parametric excitation ----------
    push!(ex, (name="17_mathieu_strong_excitation", cat="Engineering/parametric",
        prob=SDDEProblem(2, 4π, t->[0.0 1.0; -(2.0+4.0*cos(0.5*t)) -0.15],
            [(t->2π, t->[0.0 0.0; 0.6 0.0])],
            [(t->[0.0 0.0; -0.12*(2.0+4.0*cos(0.5*t)) 0.0],
              [t->[0.0 0.0; 0.072 0.0]], t->[0.0,0.0])]),
        notes="Mathieu with strong parametric excitation ε=4. Tests large periodic coeffs."))

    # ---------- 18. 3-DOF chain of delayed oscillators (higher-dim) ----------
    push!(ex, (name="18_chain_3dof", cat="Engineering/highdim",
        prob=SDDEProblem(4, 2π,
            t->[0.0 1.0 0.0 0.0; -2.0 -0.2 1.0 0.0; 0.0 0.0 0.0 1.0; 1.0 0.0 -2.0 -0.2],
            [(t->1.5, t->[0.0 0.0 0.0 0.0; -0.3 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 -0.3 0.0])],
            [(t->zeros(4,4), [t->zeros(4,4)], t->[0.0,0.1,0.0,0.1])]),
        notes="d=4: two coupled delayed oscillators (3? actually 2-DOF chain). Higher-dim, additive noise."))

    # ---------- 19. Sinusoidal (periodic) coefficient scalar, additive ----------
    push!(ex, (name="19_periodic_coeff_scalar", cat="Math/periodic",
        prob=SDDEProblem(1, 2π, t->sm1(-1.0-0.5*cos(t)), [(t->Float64(π), t->sm1(-0.3))],
            [(t->sm1(0.0), [t->sm1(0.0)], t->[0.2+0.1*sin(t)])]),
        notes="Scalar with periodic drift and periodic additive noise intensity. T=2π>τ=π."))

    # ---------- 20. Inverted pendulum with delayed stabilization (control) ----------
    push!(ex, (name="20_inverted_pendulum_delay", cat="Engineering/control",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; 9.0 0.0],   # unstable (g/l>0)
            [(t->0.3, t->[0.0 0.0; -12.0 -4.0])],          # delayed stabilizing feedback
            [(t->[0.0 0.0; 0.2 0.0], [t->zeros(2,2)], t->[0.0,0.1])]),
        notes="Inverted pendulum stabilized by delayed feedback τ=0.3. Sensorimotor noise."))

    # ---------- 21. Two-delay oscillator (engineering, multi-delay 2D) ----------
    push!(ex, (name="21_two_delay_oscillator", cat="Engineering/multi-delay",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; -3.0 -0.25],
            [(t->1.0, t->[0.0 0.0; -0.4 0.0]), (t->2.0, t->[0.0 0.0; 0.2 0.0])],
            [(t->[0.0 0.0; 0.1 0.0], [t->zeros(2,2), t->zeros(2,2)], t->[0.0,0.0])]),
        notes="2D oscillator with TWO delays τ₁=1, τ₂=2 + multiplicative noise. Multi-delay."))

    # ---------- 22. Coherence-resonance / noisy delayed bistable (physics) ----------
    push!(ex, (name="22_delayed_bistable_physics", cat="Physics",
        prob=SDDEProblem(1, 1.0, t->sm1(-0.6), [(t->1.0, t->sm1(0.5))],
            [(t->sm1(0.25), [t->sm1(0.0)], t->[0.0])]),
        notes="Linearized delayed bistable system near threshold; strong multiplicative noise."))

    # ---------- 23. Time-varying delay + periodic coeff (combined hard case) ----------
    push!(ex, (name="23_tvdelay_periodic_combined", cat="Math/combined",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; -(2.5+1.0*cos(t)) -0.2],
            [(t->1.0+0.4*sin(t), t->[0.0 0.0; 0.4 0.0])],
            [(t->[0.0 0.0; -0.08*(2.5+cos(t)) 0.0], [t->[0.0 0.0; 0.05 0.0]], t->[0.0,0.0])]),
        notes="Combined: periodic coefficients + time-varying delay τ(t)=1+0.4 sin t. Hard case."))

    # ---------- 24. Economic dynamics (Kalecki business cycle) with delay ----------
    push!(ex, (name="24_kalecki_business_cycle", cat="Economics",
        prob=SDDEProblem(2, 2π, t->[0.0 1.0; -0.5 0.2],
            [(t->1.8, t->[0.0 0.0; -0.7 0.0])],
            [(t->[0.0 0.0; 0.05 0.0], [t->zeros(2,2)], t->[0.0,0.08])]),
        notes="Kalecki/Kaldor business-cycle model: investment delay + market noise. T=2π>τ=1.8."))

    return ex
end
