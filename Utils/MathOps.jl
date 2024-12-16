module MathOps

using Random
using Statistics
using LinearAlgebra

# Softplus function
function softplus(x, beta=1.0, threshold=20.0)
    s = ifelse(x <= threshold, log(exp(beta * x) + 1.0) / beta, x)
    return s
end

# Inverse Softplus function
function inverse_softplus(s, beta=1.0, threshold=20.0)
    x = ifelse(s <= threshold, log(exp(beta * s) - 1.0) / beta, s)
    return x
end

# Covariance function
function covariance(x, y)
    return mean(x .* y, dims=1) - mean(x, dims=1) .* mean(y, dims=1)
end

end # module MathOps
