module NormalDistribution

using Distributions: logpdf, Normal, ContinuousUnivariateDistribution
import Distributions: logpdf, Normal, pdf, quantile
using ..Utils.MathOps: softplus, inverse_softplus, logsumexp
using Random
export NormalDist
import SpecialFunctions: erfinv
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
    function NormalDist(loc::T, scale::T; validate_args::Bool=false) where {T<:Real}
        new{T}(loc, positive_inverse(scale))
    end
end

function logpdf(d::NormalDist, x)
    σ = positive_function(d.optim_scale)
    return logpdf(Normal(d.loc, σ), x)
end

function params(d::NormalDist)
    return [d.loc, positive_function(d.optim_scale)]
end

function optim_params(d::NormalDist)
    return [d.loc, d.optim_scale]
end

# keeping a reference to the “inner” constructor
const _origNormalDist = NormalDist
# outer constructor: accept any Real pair, ignore kwargs, promote to Float64
function NormalDist(loc::Real, scale::Real; kwargs...)
    return _origNormalDist(float(loc), float(scale))
end


σ(d) = positive_function(d.optim_scale)

rand(rng::Random.AbstractRNG, d::NormalDist) =
    d.loc + σ(d) * randn(rng)

pdf(d::NormalDist, x) =  exp(logpdf(d, x))

quantile(d::NormalDist, p::Real) =
    d.loc + σ(d) * √2 * erfinv(2p - 1)

end # module NormalDistribution