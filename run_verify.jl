using Pkg
Pkg.activate(".")
Pkg.add("IterativeSolvers")
include("verify_mf.jl")
