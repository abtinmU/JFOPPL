module CustomGamma

using Random
using ..Utils.MathOps: softplus, inverse_softplus
using Distributions: Gamma, DiscreteMultivariateDistribution

const scaling_function = "softplus"
positive_function, positive_inverse = if scaling_function == "softplus"
    (softplus, inverse_softplus)
elseif scaling_function == "exponential"
    (exp, log)
else
    error("Scaling function not recognized")
end

struct CustomGamma{T<:Real} <: ContinuousUnivariateDistribution
    concentration::T
    optim_rate::T
    function CustomGamma(concentration::T, rate::T) where {T<:Real}
        new(concentration, positive_inverse(rate))
    end
end

function Base.logpdf(d::CustomGamma, x)
    α = positive_function(d.concentration)
    β = positive_function(d.optim_rate)
    return Distributions.logpdf(Gamma(α, β), x)
end  

function params(d::CustomGamma)
    return [positive_function(d.concentration), positive_function(d.optim_rate)]
end

function optim_params(d::CustomGamma)
    return [d.concentration, d.optim_rate]
end

end # module CustomGamma
