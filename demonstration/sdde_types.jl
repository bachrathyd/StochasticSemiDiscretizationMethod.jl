# Minimal SDDE problem specification used by the demonstration driver.
# An Itô SDDE:
#   dx = [ A(t) x + Σₖ Bₖ(t) x(t-τₖ(t)) + c(t) ] dt
#      + Σⱼ [ αⱼ(t) x + Σₖ βₖⱼ(t) x(t-τₖ(t)) + σⱼ(t) ] dWⱼ
struct SDDEProblem
    d::Int                                          # state dimension
    T::Float64                                      # principal period
    A::Function                                     # A(t) :: d×d
    delays::Vector{Tuple{Function,Function}}        # [(τₖ(t)::Float64, Bₖ(t)::d×d)]
    noise::Vector{Tuple{Function,Vector{Function},Function}}  # [(α(t), [βₖ(t)], σ(t))]
end

maxdelay(prob::SDDEProblem, ts) =
    maximum(maximum(τ(t) for t in ts) for (τ,_) in prob.delays)
