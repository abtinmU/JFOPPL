module CustomCategorical

using Random
using Distributions: Categorical
using StatsFuns: logsumexp

struct CustomCategorical{T<:AbstractVector} <: DiscreteMultivariateDistribution
    logits::T
    function CustomCategorical(probs::T) where {T<:AbstractVector}
        logits = logits = log.(probs) .- logsumexp(log.(probs))
        new(logits .- logsumexp(logits))
    end
end

function Base.logpdf(d::CustomCategorical, x)
    return logpdf(Categorical(logits=d.logits), x)
end

function params(d::CustomCategorical)
    return [d.logits]
end

function optim_params(d::CustomCategorical)
    return [d.logits]
end

end # module CustomCategorical
