module GibbsSampling

using Random
using Statistics
using LinearAlgebra
using Printf
using .GraphBasedSamplingUtils: Graph, evaluate_graph, split_nodes_into_sample_observe, evaluate_node
using ..Utils.SampleHandling: burn_chain, flatten_sample

export Gibbs_samples

### MH within Gibbs ###

function Gibbs_samples(g::Graph, num_samples; tmax=nothing, burn_frac=nothing, wandb_name=nothing, debug=false, verbose=false)
    sample_nodes, _ = split_nodes_into_sample_observe(g)

    samples, weights = [], []
    accepted_small_steps = 0; num_small_steps = 0; num_big_steps = 0
    max_time = tmax !== nothing ? time() + tmax : nothing

    for i in 1:num_samples
        if i == 1
            result, sig, l = evaluate_graph(g, verbose=verbose)
        else
            for resample_node in sample_nodes
                resample_logP = l[resample_node * "_logP"]
                sig_here, l_here = deepcopy(sig), deepcopy(l)
                d = eval(g.expressions[resample_node], sig_here, l_here)
                resample_logP_new = sig_here["logP"] - sig["logP"]
                fixed_nodes, fixed_probs = Dict(resample_node => d), Dict(resample_node => resample_logP_new)
                if debug
                    println("Original node value: ", l[resample_node])
                    println("Original node logP: ", resample_logP)
                    println("Resampled node value: ", d)
                    println("Resampled node logP: ", resample_logP_new)
                end

                for node in g.nodes
                    if node != resample_node
                        fixed_nodes[node] = l[node]
                        if !(node in g.arrows[resample_node])
                            fixed_probs[node] = l[node * "_logP"]
                        end
                    end
                end
                if debug
                    println("Fixed nodes: ", fixed_nodes)
                    println("Fixed probabilities: ", fixed_probs)
                end
                result_new, sig_new, l_new = evaluate_graph(g, fixed_nodes=fixed_nodes, fixed_probs=fixed_probs, verbose=verbose)
                if debug
                    println("Old sig: ", sig)
                    println("New sig: ", sig_new)
                    println("Old environment: ", l)
                    println("New environment: ", l_new)
                end

                acceptance = exp(sig_new["logP"] - sig["logP"] - resample_logP_new + resample_logP)
                alpha = min(1.0, acceptance)
                accept = rand() < alpha
                if accept
                    result, sig, l = result_new, sig_new, l_new
                    accepted_small_steps += 1
                end
                if wandb_name !== nothing
                    log_sample(result, i, wandb_name)
                end
                num_small_steps += 1
                if debug
                    break
                end
            end
            if debug
                break
            end
        end

        num_big_steps += 1
        push!(samples, result)
        push!(weights, 1.0)
        if tmax !== nothing && time() > max_time
            break
        end
    end

    @printf("Acceptance fraction: %.3f\n", accepted_small_steps / num_small_steps)
    println("Number of samples: ", num_big_steps)
    if burn_frac !== nothing
        println("Burn fraction: ", burn_frac)
        nburn = Int(burn_frac * num_big_steps)
        println("Burning up to: ", nburn)
        samples, weights = burn_chain(samples, weights, burn_frac)
    end

    return samples, weights
end

end # module GibbsSampling
