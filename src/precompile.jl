using PrecompileTools: @setup_workload, @compile_workload

# Exercise the main CPU entry points once at build time so the first real call
# in a user session is fast. Kept intentionally tiny (a 2-state delay oscillator,
# very few steps) — the goal is to cache method specializations, not to compute.
@setup_workload begin
    A = ProportionalMX(@SMatrix [0.0 1.0; -1.0 -0.2])
    B = DelayMX(2π, @SMatrix [0.0 0.0; 0.1 0.0])
    α = stCoeffMX(1, ProportionalMX(@SMatrix [0.0 0.0; 0.05 0.0]))
    β = stCoeffMX(1, DelayMX(2π, @SMatrix [0.0 0.0; 0.02 0.0]))
    σ = stAdditive(1, Additive(@SVector [0.0, 0.5]))
    prob = LDDEProblem(A, [B], [α], [β], Additive(2), [σ])
    @compile_workload begin
        rst = calculateResults(prob, SemiDiscretization(1, 2π / 6), 2π;
                               n_steps = 6, calculate_additive = true)
        spectralRadiusOfMapping_MF_factored(rst)
        fixPointOfMapping_MF_factored(rst)
        dm = DiscreteMapping_M2_MF(rst)
        spectralRadiusOfMapping_MF(dm)
        fixPointOfMapping_MF(dm)
        m2 = DiscreteMapping_M2(rst)
        spectralRadiusOfMapping(m2)
        fixPointOfMapping(m2)
    end
end
