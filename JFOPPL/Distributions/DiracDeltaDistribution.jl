module DiracDeltaDistribution

using Random
using Distributions

# Type representing a delta distribution
struct DeltaDistribution
    x::Any
end

# Sampling from the delta distribution always returns the same value
function sample(dist::DeltaDistribution)
    return dist.x
end

# Log probability for the delta distribution
function Base.logpdf(d::DeltaDistribution, x)
    return d.x == x ? Inf : -Inf
end 

# Function to return different types of distributions based on the scheme
function dirac_delta_distribution(x...; scheme="normal")
    if scheme == "normal"
        return Normal(x[1], 0.1)
    elseif scheme == "uniform"
        return Uniform(x[1] - 0.05, x[1] + 0.05)
    elseif scheme == "delta"
        return DeltaDistribution(x[1])
    else
        error("Dirac delta scheme not recognized")
    end
end

end # module DiracDeltaDistribution
