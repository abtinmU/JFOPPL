module NormalDistribution

using Random
using ..Utils.MathOps: softplus, inverse_softplus
using Distributions: Normal, ContinuousUnivariateDistribution

const scaling_function = "softplus"
positive_function, positive_inverse = if scaling_function == "softplus"
    (softplus, inverse_softplus)
elseif scaling_function == "exponential"
    (exp, log)
else
    error("Scaling function not recognized")
end

struct NormalDist{T<:Real} <: ContinuousUnivariateDistribution
    loc::T
    optim_scale::T
    function NormalDist(loc::T, scale::T) where {T<:Real}
        new(loc, positive_inverse(scale))
    end
end

function Base.logpdf(d::NormalDist, x)
    σ = positive_function(d.optim_scale)
    return Distributions.logpdf(Distributions.Normal(d.loc, σ), x)
end

function params(d::NormalDist)
    return [d.loc, positive_function(d.optim_scale)]
end

function optim_params(d::NormalDist)
    return [d.loc, d.optim_scale]
end

end # module NormalDistribution
