module CustomDirichlet

using Random
using ..Utils.MathOps: softplus, inverse_softplus
using Distributions: Dirichlet

const scaling_function = "softplus"
positive_function, positive_inverse = if scaling_function == "softplus"
    (softplus, inverse_softplus)
elseif scaling_function == "exponential"
    (exp, log)
else
    error("Scaling function not recognized")
end

struct CustomDirichlet{T<:AbstractVector} <: ContinuousMultivariateDistribution
    optim_concentration::T
    function CustomDirichlet(concentration::T) where {T<:AbstractVector}
        new(positive_inverse.(concentration))
    end
end

function Base.logpdf(d::CustomDirichlet, x)
    α = positive_function.(d.optim_concentration)
    return Distributions.logpdf(Dirichlet(α), x)
end  

function params(d::CustomDirichlet)
    return [positive_function.(d.optim_concentration)]
end

function optim_params(d::CustomDirichlet)
    return d.optim_concentration
end

end # module CustomDirichlet
