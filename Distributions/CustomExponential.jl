module CustomExponential

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

struct CustomExponential{T<:Real} <: ContinuousUnivariateDistribution
    optim_rate::T
    function CustomExponential(rate::T) where {T<:Real}
        new(positive_inverse(rate))
    end
end

function Base.logpdf(d::CustomExponential, x)
    rate = positive_function(d.optim_rate)
    return logpdf(Exponential(rate), x)
end

function params(d::CustomExponential)
    return [positive_function(d.optim_rate)]
end

function optim_params(d::CustomExponential)
    return [d.optim_rate]
end

end # module CustomExponential
