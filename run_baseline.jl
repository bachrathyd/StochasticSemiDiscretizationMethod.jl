using Pkg
Pkg.instantiate()
Pkg.add(["BenchmarkTools", "Plots", "Printf"])
include("work_precision_original.jl")
