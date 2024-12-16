module GeneralSampling

using Random
using Statistics
using LinearAlgebra
using Printf
using .EvaluationBasedSampling: evaluate_program
using .GraphBasedSamplingUtils: evaluate_graph, burn_chain, log_sample, flatten_sample

export get_sample, prior_samples, calculate_effective_sample_size, resample_using_importance_weights, Metropolis_Hastings_samples

function get_sample(ast_or_graph, mode::String; verbose=false)
    if mode == "desugar"
        ret, sig, _ = evaluate_program(ast_or_graph, verbose=verbose)
    elseif mode == "graph"
        ret, sig, _ = evaluate_graph(ast_or_graph, verbose=verbose)
    else
        error("Mode not recognised")
    end
    ret = flatten_sample(ret)
    return ret, sig
end

function prior_samples(ast_or_graph, mode::String, num_samples::Int; tmax=nothing, wandb_name=nothing, verbose=false)
    samples, weights = [], []
    max_time = tmax !== nothing ? time() + tmax : nothing
    for i in 1:num_samples
        sample, sig = get_sample(ast_or_graph, mode, verbose)
        weight = sig["logW"]
        if wandb_name !== nothing
            log_sample(sample, i, wandb_name=wandb_name)
        end
        push!(samples, sample)
        push!(weights, weight)
        if tmax !== nothing && time() > max_time
            break
        end
    end
    return samples, weights
end

function calculate_effective_sample_size(weights::Vector; verbose=false)
    weights /= sum(weights)
    ESS = 1.0 / sum(weights .^ 2)
    if verbose
        println("Effective sample size: ", ESS)
        println("Fractional sample size: ", ESS / length(weights))
        println("Sum of weights: ", sum(weights))
    end
    return ESS
end

function resample_using_importance_weights(samples, log_weights; normalize=true, wandb_name=nothing)
    nsamples = size(samples, 1)
    if normalize
        log_weights = log_weights .- maximum(log_weights)
    end
    weights = exp.(log_weights)
    ESS = calculate_effective_sample_size(weights, verbose=true)
    indices = sample(1:nsamples, Weights(weights), nsamples)
    new_samples = samples[indices, :]
    if wandb_name !== nothing
        for (i, sample) in enumerate(new_samples)
            log_sample(sample, i, wandb_name, resample=true)
        end
    end
    return new_samples
end

function Metropolis_Hastings_samples(ast_or_graph, mode::String, num_samples::Int; tmax=nothing, burn_frac=nothing, wandb_name=nothing, verbose=false)
    accepted_steps = 0
    num_steps = 0
    samples, weights = [], []
    max_time = tmax !== nothing ? time() + tmax : nothing
    old_sample, old_prob = nothing, nothing
    for i in 1:num_samples
        sample, sig = get_sample(ast_or_graph, mode, verbose)
        prob = exp(sig["logW"])
        if i != 1
            acceptance = min(1.0, prob / old_prob)
            accept = rand() < acceptance
        else
            accept = true
        end
        if accept
            new_sample = sample
            new_prob = prob
            accepted_steps += 1
        else
            new_sample = old_sample
            new_prob = old_prob
        end
        num_steps += 1
        if wandb_name !== nothing
            log_sample(sample, i, wandb_name)
        end
        push!(samples, new_sample)
        push!(weights, 1.0)
        old_sample, old_prob = new_sample, new_prob
        if tmax !== nothing && time() > max_time
            break
        end
    end
    println("Acceptance fraction: ", accepted_steps / num_steps)
    samples, weights = burn_chain(samples, weights, burn_frac=burn_frac)
    return samples, weights
end

end # module GeneralSampling
