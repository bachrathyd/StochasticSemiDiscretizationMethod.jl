```@meta
CurrentModule = StochasticSemiDiscretizationMethod
```

# API reference

```@index
```

## Problem definition

```@docs
LDDEProblem
stCoeffMX
stAdditive
```

## Moment maps

```@docs
DiscreteMapping_M1
DiscreteMapping_M2
DiscreteMapping_M2_MF
```

## Stability and stationary moments

```@docs
spectralRadiusOfMapping
fixPointOfMapping
spectralRadiusOfMapping_MF
fixPointOfMapping_MF
spectralRadiusOfMapping_MF_factored
fixPointOfMapping_MF_factored
```

## [GPU backend](@id gpu)

The GPU methods live in a package extension that is loaded automatically the
moment `using CUDA` is run alongside this package (and a functional CUDA device
is available). Without CUDA loaded, [`spectralRadiusOfMapping_auto`](@ref) falls
back to the CPU solver and the `*_GPU` entry points raise an informative error.

```@docs
spectralRadiusOfMapping_GPU
fixPointOfMapping_GPU
spectralRadiusOfMapping_auto
```

## Covariance utilities

```@docs
MxToCovVec
VecToCovMx
```
