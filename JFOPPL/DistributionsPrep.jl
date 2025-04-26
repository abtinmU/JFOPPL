module DistributionsPrep

using Distributions: Normal, Beta, Exponential, Uniform, Categorical, Bernoulli, Gamma, Dirichlet
using StatsFuns: logsumexp

using Random

# Import custom distributions
using .CustomDistributions.NormalDistribution: NormalDist
using .CustomDistributions.CustomGamma: GammaDist
using .CustomDistributions.CustomBeta: BetaDist
using .CustomDistributions.CustomExponential: ExponentialDist
using .CustomDistributions.CustomDirichlet: DirichletDist
using .CustomDistributions.CustomCategorical: CategoricalDist
using .CustomDistributions.CustomBernoulli: BernoulliDist
using .CustomDistributions.DiracDeltaDistribution: dirac_delta_distribution

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
    "normal" => NormalDist,
    "beta" => BetaDist,
    "exponential" => ExponentialDist,
    "uniform-continuous" => Uniform,
    "discrete" => CategoricalDist,
    "bernoulli" => BernoulliDist,
    "gamma" => GammaDist,
    "dirichlet" => DirichletDist,
    "flip" => BernoulliDist,
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