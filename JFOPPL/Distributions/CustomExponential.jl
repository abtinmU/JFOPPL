module CustomExponential

using Random
using ..Utils.MathOps: softplus, inverse_softplus
using Distributions: Exponential

const scaling_function = "softplus"
positive_function, positive_inverse = if scaling_function == "softplus"
    (softplus, inverse_softplus)
elseif scaling_function == "exponential"
    (exp, log)
else
    error("Scaling function not recognized")
end

struct CustomExponential{T<:Real} <: ContinuousUnivariateDistribution
    optim_rate::T
    function CustomExponential(rate::T) where {T<:Real}
        new(positive_inverse(rate))
    end
end

function Base.logpdf(d::CustomExponential, x)
    λ = positive_function(d.optim_rate)
    return Distributions.logpdf(Exponential(λ), x)
end  

function params(d::CustomExponential)
    return [positive_function(d.optim_rate)]
end

function optim_params(d::CustomExponential)
    return [d.optim_rate]
end

end # module CustomExponential
