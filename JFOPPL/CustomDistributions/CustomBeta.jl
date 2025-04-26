module CustomBeta

using Random
using ..Utils.MathOps: softplus, inverse_softplus
using Distributions: Beta, DiscreteMultivariateDistribution

const scaling_function = "softplus"
positive_function, positive_inverse = if scaling_function == "softplus"
    (softplus, inverse_softplus)
elseif scaling_function == "exponential"
    (exp, log)
else
    error("Scaling function not recognized")
end

struct CustomBeta{T<:Real} <: ContinuousUnivariateDistribution
    optim_concentration1::T
    optim_concentration0::T
    function CustomBeta(concentration1::T, concentration0::T) where {T<:Real}
        new(positive_inverse(concentration1), positive_inverse(concentration0))
    end
end

function Base.logpdf(d::CustomBeta, x)
    α = positive_function(d.optim_concentration1)
    β = positive_function(d.optim_concentration0)
    return Distributions.logpdf(Beta(α, β), x)
end
  
function params(d::CustomBeta)
    return [positive_function(d.optim_concentration1), positive_function(d.optim_concentration0)]
end

function optim_params(d::CustomBeta)
    return [d.optim_concentration1, d.optim_concentration0]
end

end # module CustomBeta
