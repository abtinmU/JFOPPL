module CustomBernoulli

using Distributions: Bernoulli
using Random

struct CustomBernoulli{T<:Real} <: DiscreteUnivariateDistribution
    logits::T
    function CustomBernoulli(probs::T) where {T<:Real}
        new(log(probs / (1 - probs)))
    end
end

function Base.logpdf(d::CustomBernoulli, x)
    return Distributions.logpdf(Bernoulli(logits=d.logits), x)
end

function params(d::CustomBernoulli)
    return [d.logits]
end

function optim_params(d::CustomBernoulli)
    return [d.logits]
end

end # module CustomBernoulli
