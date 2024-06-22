module NormalDistribution

using Random
using .Utils: softplus, inverse_softplus

const scaling_function = "softplus"
positive_function, positive_inverse = if scaling_function == "softplus"
    (softplus, inverse_softplus)
elseif scaling_function == "exponential"
    (exp, log)
else
    error("Scaling function not recognized")
end

struct Normal{T<:Real} <: ContinuousUnivariateDistribution
    loc::T
    optim_scale::T
    function Normal(loc::T, scale::T) where {T<:Real}
        new(loc, positive_inverse(scale))
    end
end

function Base.logpdf(d::Normal, x)
    scale = positive_function(d.optim_scale)
    return logpdf(Normal(d.loc, scale), x)
end

function params(d::Normal)
    return [d.loc, positive_function(d.optim_scale)]
end

function optim_params(d::Normal)
    return [d.loc, d.optim_scale]
end

end # module NormalDistribution
