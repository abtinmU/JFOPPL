module DistributionsPrep

using Random

# Import custom distributions from the Distributions folder
using .Distributions.NormalDistribution
using .Distributions.CustomGamma
using .Distributions.CustomExponential
using .Distributions.CustomBeta
using .Distributions.CustomDirichlet
using .Distributions.CustomBernoulli
using .Distributions.CustomCategorical
using .Distributions.DiracDeltaDistribution: dirac_delta_distribution

# List of all supported distributions
distributions = [
    "normal",
    "beta",
    "exponential",
    "uniform-continuous",
    "discrete",
    "bernoulli",
    "gamma",
    "dirichlet",
    "flip",
    "dirac"
]

# Dictionary mapping distribution names to Julia distribution constructors
distribution_constructors = Dict(
    "normal" => Normal,
    "beta" => CustomBeta,
    "exponential" => CustomExponential,
    "uniform-continuous" => Uniform,
    "discrete" => CustomCategorical,
    "bernoulli" => CustomBernoulli,
    "gamma" => CustomGamma,
    "dirichlet" => CustomDirichlet,
    "flip" => CustomBernoulli,
    "dirac" => dirac_delta_distribution
)

# Starting parameters for distributions for necessary cases
distribution_params = Dict(
    "normal-params" => (0.0, 1.0),
    "beta-params" => (1.0, 1.0),
    "exponential-params" => (1.0,),
    "uniform-continuous-params" => (0.0, 1.0),
    "discrete-params" => ([1/3, 1/3, 1/3],),
    "bernoulli-params" => (0.5,),
    "gamma-params" => (1.0, 1.0),
    "dirichlet-params" => ([1.0, 1.0, 1.0],),
    "flip-params" => (0.5,)
)

end # module DistributionsPrep
