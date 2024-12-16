module SampleHandling

using Random
using Statistics
using LinearAlgebra

# Flatten sample function
function flatten_sample(sample)
    if isa(sample, Array{Array})
        flat_sample = vcat([vec(element) for element in sample]...)
    else
        flat_sample = sample
    end
    return flat_sample
end

# Create unique list function
function create_unique_list(list_with_duplicates)
    return collect(OrderedDict(zip(list_with_duplicates, 1:length(list_with_duplicates))))
end

# Burn chain function
function burn_chain(samples, weights, burn_frac=nothing)
    if burn_frac !== nothing
        n = length(samples)
        nburn = Int(burn_frac * n)
        burned_samples = samples[nburn+1:end]
        burned_weights = weights[nburn+1:end]
        return burned_samples, burned_weights
    else
        return samples, weights
    end
end

end # module SampleHandling
