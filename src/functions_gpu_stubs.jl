# CPU-side fallbacks for the GPU entry points. The real device methods live in
# the CUDA package extension (ext/StochasticSemiDiscretizationMethodCUDAExt.jl)
# and become available the moment the user runs `using CUDA` next to this
# package. These generic fallbacks keep the exported names always defined:
#   * `spectralRadiusOfMapping_auto` transparently uses the CPU MF solver;
#   * the `*_GPU` entry points raise an informative error until CUDA is loaded.
# The extension methods are typed on `stDiscreteMapping_M2_MF`, hence strictly
# more specific than these `::Any` fallbacks, so dispatch prefers them whenever
# the CUDA extension is active.

_needs_cuda(fname) = error(
    "`$fname` requires the GPU backend, which lives in a package extension. " *
    "Run `using CUDA` (with a functional CUDA device) to enable it.")

"""
    spectralRadiusOfMapping_GPU(dm::DiscreteMapping_M2_MF; krylovdim=0, kwargs...)

Second-moment spectral radius ``\\rho(\\mathcal{H})`` evaluated on the GPU with
the multiplication-free operator: a matrix-free Arnoldi iteration that runs
entirely on the device, transferring back only the dominant eigenvalue.

Only available when the CUDA extension is loaded (`using CUDA`) and a functional
CUDA device is present; otherwise an informative error is raised. See also
[`spectralRadiusOfMapping_MF`](@ref) (CPU) and
[`spectralRadiusOfMapping_auto`](@ref) (automatic CPU/GPU selection).
"""
spectralRadiusOfMapping_GPU(::Any; kwargs...) = _needs_cuda("spectralRadiusOfMapping_GPU")

"""
    fixPointOfMapping_GPU(dm::DiscreteMapping_M2_MF; rtol=1e-15, kwargs...)

Stationary second-moment fixed point ``\\mathbf{M}^\\ast`` evaluated on the GPU
with the multiplication-free operator and an on-device Krylov linear solve.

Only available when the CUDA extension is loaded (`using CUDA`); otherwise an
informative error is raised. See also [`fixPointOfMapping_MF`](@ref) (CPU).
"""
fixPointOfMapping_GPU(::Any; kwargs...) = _needs_cuda("fixPointOfMapping_GPU")

"""
    spectralRadiusOfMapping_auto(dm::DiscreteMapping_M2_MF; cpu_threshold=10_000, krylovdim=0, kwargs...)

Automatically choose the CPU or GPU multiplication-free solver for the
second-moment spectral radius, switching to the GPU when the covariance
dimension ``D`` exceeds `cpu_threshold` and a functional CUDA device is present.

Without the CUDA extension loaded this always uses the CPU path
([`spectralRadiusOfMapping_MF`](@ref)); loading `using CUDA` activates the
device-aware method.
"""
spectralRadiusOfMapping_auto(dm; kwargs...) = spectralRadiusOfMapping_MF(dm; kwargs...)
