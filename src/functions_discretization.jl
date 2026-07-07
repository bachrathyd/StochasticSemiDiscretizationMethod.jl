function calculateResults(LDDEP::LDDEProblem, method::DiscretizationMethod, DiscretizationLength::Real; n_steps::Int64 = nStepOfLength(DiscretizationLength, method.Δt), calculate_additive::Bool = false, im::ItoIsometryMethod = Trapezoidal(20, method))
    result = Result(LDDEP, method, DiscretizationLength, n_steps = n_steps, calculate_additive = calculate_additive, im = im)
    calculateDetResults!(result)

    calculateStResults!(result)
    return (result)
end

function calculateStResults!(rst::AbstractResult)
    registerStResult!(rst,
    [rst.method(mx, rst) for mx in [rst.LDDEP.αs...,rst.LDDEP.βs...]])
    if rst.calculate_additive
        calculateStAdditiveResults!(rst)
    end
end
function calculateStAdditiveResults!(rst::AbstractResult)
    registerStAdditiveResults!(rst,
        rst.method.(rst.LDDEP.σs, Ref(rst)))
end
function registerStResult!(rst::AbstractResult, stsubmxs::Vector{<:Vector{<:stSubMX}})
    rst.stsubMXs .= stsubmxs;
end

function registerStAdditiveResults!(rst::AbstractResult, subvs::Vector{<:Vector{<:stSubV}})
    rst.stsubVs .= subvs;
end

function stDiscreteMappingSteps(rst::AbstractResult{d}) where d
    ns = calculate_noise_mxelems(rst)
    Iss = [[Vector{Int64}(undef, n) for n in ns] for x in 1:rst.n_steps]
    Jss = [[Vector{Int64}(undef, n) for n in ns] for x in 1:rst.n_steps]
    Vss = [[Vector{eltype(rst.stsubMXs[1][1].MXfun[1])}(undef, n) for n in ns] for x in 1:rst.n_steps]
    n0 = [fill(1, length(ns)) for i in 1:rst.n_steps]
    n1 = [zero(ns) for i in 1:rst.n_steps]
    for stsmx in rst.stsubMXs
        w = stsmx[1].nID
        for t in 1:rst.n_steps
            for i in eachindex(stsmx[t].ranges, stsmx[t].MXfun)
                n1[t][w] = n0[t][w] + d^2 - 1
                IJs = Iterators.product(stsmx[t].ranges[i]...) |> collect |> vec
                Iss[t][w][n0[t][w]:n1[t][w]] .= getindex.(IJs, 1)
                Jss[t][w][n0[t][w]:n1[t][w]] .= getindex.(IJs, 2)
                Vss[t][w][n0[t][w]:n1[t][w]] .= stsmx[t].MXfun[i][:]
                n0[t][w] = n1[t][w] + 1
            end
        end
    end
    if rst.calculate_additive
        ns = calculate_noise_velems(rst)
        IssV = [[Vector{Int64}(undef, n) for n in ns] for x in 1:rst.n_steps]
        VssV = [[Vector{eltype(rst.stsubVs[1][1].Vfun)}(undef, n) for n in ns] for x in 1:rst.n_steps]
        n0 = [fill(1, length(ns)) for i in 1:rst.n_steps]
        n1 = [zero(ns) for i in 1:rst.n_steps]
        for stsv in rst.stsubVs
            w = stsv[1].nID
            for t in 1:rst.n_steps
                n1[t][w] = n0[t][w] + d - 1
                IssV[t][w][n0[t][w]:n1[t][w]] .= Base.OneTo(d)
                VssV[t][w][n0[t][w]:n1[t][w]] .= stsv[t].Vfun
                n0[t][w] = n1[t][w] + 1
            end
        end
    else
        IssV = fill(fill(Vector{Int64}(undef, 0), length(ns)), rst.n_steps)
        VssV = fill(fill(Vector{Vector{Float64}}(undef, 0), length(ns)), rst.n_steps)
    end
    ([[sparse(Iss[t][w], Jss[t][w], Vss[t][w], rst.n, rst.n) for w in 1:rst.LDDEP.w] for t in 1:rst.n_steps], [[sparsevec(IssV[t][w], VssV[t][w], rst.n) for w in 1:rst.LDDEP.w] for t in 1:rst.n_steps]) # stMXs, stVs
end

function stDiscreteMapping(rst::AbstractResult)
    stDiscreteMapping(DiscreteMappingSteps(rst)...,
        stDiscreteMappingSteps(rst)...)
end

"""
    DiscreteMapping_M1(rst::AbstractResult[, idxs])
    DiscreteMapping_M1(prob::LDDEProblem, method, DiscretizationLength; n_steps, calculate_additive=false)

Assemble the one-period **first-moment** (mean) map of the delay problem — the
deterministic semi-discretization monodromy. Its [`spectralRadiusOfMapping`](@ref)
is the deterministic Floquet spectral radius and its [`fixPointOfMapping`](@ref)
the stationary mean.
"""
DiscreteMapping_M1(rst::AbstractResult) = DiscreteMapping_M1(DiscreteMappingSteps(rst)...)
DiscreteMapping_M1(rst::AbstractResult, idxs::AbstractArray{<:Integer}) = DiscreteMapping_M1(DiscreteMappingSteps(rst)..., idxs)
function DiscreteMapping_M1(LDDEP::LDDEProblem, method::DiscretizationMethod, DiscretizationLength::Real; args...)
    DiscreteMapping_M1(calculateResults(LDDEP, method, DiscretizationLength; args...))
end
function DiscreteMapping_M1(LDDEP::LDDEProblem, method::DiscretizationMethod, DiscretizationLength::Real, idxs; args...)
    DiscreteMapping_M1(calculateResults(LDDEP, method, DiscretizationLength; args...), idxs)
end


function DiscreteMapping_M2(stdm::stDiscreteMapping, rst::AbstractResult)
    M2_MXs = M2_Mapping_from_Sparse.(stdm.detMXs, Ref(rst)) .+
        [sum(M2_Mapping_from_Sparse(stMX, rst) for stMX in stMXsₜ) for stMXsₜ in stdm.stMXs] # H [time]
    M1toM2_MXs = fill(spzeros(eltype(eltype(M2_MXs)), size(M2_MXs, 1), rst.n), rst.n_steps)
    M2_Vs = fill(spzeros(eltype(eltype(M2_MXs)), size(M2_MXs, 1)), rst.n_steps)
    if rst.calculate_additive
        M1toM2_MXs .= M1toM2_Mapping_Generator_from_Sparse.(stdm.detMXs, stdm.detVs, Ref(rst)) .+
            [sum(M1toM2_Mapping_Generator_from_Sparse(stMX, stdm.stVs[t][i], rst) for (i, stMX) in enumerate(stMXsₜ)) for (t, stMXsₜ) in enumerate(stdm.stMXs)]
        M2_Vs .= M2_Additive_from_Sparse.(stdm.detVs, Ref(rst)) +
            [sum(M2_Additive_from_Sparse(stV, rst) for stV in stVₜ) for stVₜ in stdm.stVs]
    end
    DiscreteMapping_M2(stdm.ts, stdm.detMXs, stdm.detVs, M2_MXs, M1toM2_MXs, M2_Vs)
end

function DiscreteMapping_M2(stdm::stDiscreteMapping, idxs::AbstractArray{<:Integer}, rst::AbstractResult)
    M2_MXs = M2_Mapping_from_Sparse.(stdm.detMXs, Ref(idxs), Ref(rst))
    for t in eachindex(M2_MXs)
        M2_MXs[t] += sum(M2_Mapping_from_Sparse(stMX, idxs, rst) for stMX in stdm.stMXs[t])  # H [time]
    end
    M1toM2_MXs = fill(spzeros(eltype(eltype(M2_MXs)), size(M2_MXs, 1), length(idxs)), rst.n_steps)
    M2_Vs = fill(spzeros(eltype(eltype(M2_MXs)), size(M2_MXs, 1)), rst.n_steps)
    if rst.calculate_additive
        M1toM2_MXs .= M1toM2_Mapping_Generator_from_Sparse.(stdm.detMXs, stdm.detVs, Ref(idxs), Ref(rst)) .+
            [sum(M1toM2_Mapping_Generator_from_Sparse(stMX, stdm.stVs[t][i], idxs, rst) for (i, stMX) in enumerate(stMXsₜ)) for (t, stMXsₜ) in enumerate(stdm.stMXs)]
        M2_Vs .= M2_Additive_from_Sparse.(stdm.detVs, Ref(idxs), Ref(rst)) .+
            [sum(M2_Additive_from_Sparse(stV, idxs, rst) for stV in stVₜ) for stVₜ in stdm.stVs]
    end
    DiscreteMapping_M2(stdm.ts, getindex.(stdm.detMXs, Ref(idxs), Ref(idxs)), getindex.(stdm.detVs, Ref(idxs)), M2_MXs, M1toM2_MXs, M2_Vs)
end

"""
    DiscreteMapping_M2(rst::AbstractResult[, idxs])
    DiscreteMapping_M2(prob::LDDEProblem, method, DiscretizationLength; n_steps, calculate_additive=false)

Assemble the **explicit** one-period second-moment (covariance) map of a
stochastic delay problem: the classical semi-discretization operator, built and
stored as a matrix. This is the reference implementation consumed by
[`spectralRadiusOfMapping`](@ref) and [`fixPointOfMapping`](@ref); it is
algebraically identical to, but ``\\mathcal{O}(p^2)`` more expensive in memory
than, the matrix-free [`DiscreteMapping_M2_MF`](@ref). Prefer the MF variants for
anything but small `p`. Pass `calculate_additive = true` for the stationary
variance. The optional `idxs` restricts the map to a subset of state components.
"""
DiscreteMapping_M2(rst::AbstractResult) = DiscreteMapping_M2(stDiscreteMapping(rst), rst)
DiscreteMapping_M2(rst::AbstractResult, idxs::AbstractArray{<:Integer}) = DiscreteMapping_M2(stDiscreteMapping(rst), idxs, rst)
function DiscreteMapping_M2(LDDEP::LDDEProblem, method::DiscretizationMethod, DiscretizationLength::Real; args...)
    DiscreteMapping_M2(calculateResults(LDDEP, method, DiscretizationLength; args...))
end
function DiscreteMapping_M2(LDDEP::LDDEProblem, method::DiscretizationMethod, DiscretizationLength::Real, idxs; args...)
    DiscreteMapping_M2(calculateResults(LDDEP, method, DiscretizationLength; args...), idxs)
end

# Some attributes of the mappings
function fixPointOfMapping(dm::DiscreteMapping_M1)
    (I - prodl(dm.M1_MXs)) \ Vector(reduce_additive(dm.M1_MXs, dm.M1_Vs))
end
function fixPointOfMapping(dm::DiscreteMapping_M1, idxs::AbstractVector{<:Integer})
    (I - prodl(dm.M1_MXs,idxs)) \ Vector(reduce_additive(dm.M1_MXs, dm.M1_Vs, idxs))
end

"""
    fixPointOfMapping(dm::DiscreteMapping_M2) -> Vector
    fixPointOfMapping(dm::DiscreteMapping_M1) -> Vector

Stationary fixed point of a discrete map. For a first-moment map
([`DiscreteMapping_M1`](@ref)) this is the stationary mean; for a second-moment
map ([`DiscreteMapping_M2`](@ref)) it is the stationary covariance in
half-vectorized coordinates (reshape with [`VecToCovMx`](@ref)). Requires the
map to have been built with `calculate_additive = true` and to be
(mean-square) stable.
"""
function fixPointOfMapping(dm::DiscreteMapping_M2)
    v1st = (I - prodl(dm.M1_MXs)) \ Vector(reduce_additive(dm.M1_MXs, dm.M1_Vs))
    (I - prodl(dm.M2_MXs)) \ Vector(reduce_additive(dm.M2_MXs, dm.M1toM2_MXs, dm.M1_MXs, dm.M2_Vs, dm.M1_Vs, v1st))
end

function spectralRadiusOfMapping(dm::DiscreteMapping_M1; args...)
    abs(eigs(prodl(dm.M1_MXs); args...)[1][1])
end
function spectralRadiusOfMapping(dm::DiscreteMapping_M1, idxs::AbstractVector{<:Integer}; args...)
    abs(eigs(prodl(dm.M1_MXs, idxs); args...)[1][1])
end

"""
    spectralRadiusOfMapping(dm::DiscreteMapping_M2; kwargs...) -> Float64
    spectralRadiusOfMapping(dm::DiscreteMapping_M1; kwargs...) -> Float64

Spectral radius of a discrete map's monodromy. For [`DiscreteMapping_M1`](@ref)
this is the deterministic (first-moment) Floquet spectral radius; for
[`DiscreteMapping_M2`](@ref) it is the second-moment spectral radius
``\\rho(\\mathcal{H})``, whose value below `1` certifies mean-square stability.
Keyword arguments are forwarded to the Arpack eigensolver. For large step counts
use the matrix-free [`spectralRadiusOfMapping_MF`](@ref) instead.
"""
function spectralRadiusOfMapping(dm::DiscreteMapping_M2; args...)
    abs(eigs(prodl(dm.M2_MXs); args...)[1][1])
end
