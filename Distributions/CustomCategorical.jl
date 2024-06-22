module CustomCategorical

using Random

struct CustomCategorical{T<:AbstractVector} <: DiscreteMultivariateDistribution
    logits::T
    function CustomCategorical(probs::T) where {T<:AbstractVector}
        logits = log.(probs / (1 .- probs))
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
