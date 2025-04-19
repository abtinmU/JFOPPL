module GraphBasedSamplingUtils

using Random
using Statistics
using LinearAlgebra
using Printf
using DataStructures: TopologicalSorter
using .BasicPrimitives: primitives
using .EvaluationBasedSampling: eval
using Zygote: gradient
using Flux.Optimise: Adam, update!
using Flux: params

export Graph, evaluate_node, evaluate_graph, generate_IC, log_joint

# Define the Graph type
struct Graph
    functions::Vector{Any}
    nodes::Vector{Any}
    arrows::Dict{Any, Any}
    expressions::Dict{Any, Any}
    observe::Any
    program::Any

    function Graph(graph_json::Vector{Any})
        g = new(
            graph_json[1],
            graph_json[2]["V"],
            graph_json[2]["A"],
            graph_json[2]["P"],
            graph_json[2]["Y"],
            graph_json[3]
        )
        g.nodes = topological_sort(g)
        return g
    end
end

function topological_sort(g::Graph; verbose=false)
    for node in g.nodes
        if !haskey(g.arrows, node)
            g.arrows[node] = []
        end
    end
    if verbose println("arrows: ", g.arrows) end

    sorter = TopologicalSorter(g.arrows)
    sorted_list = collect(sorter)
    return reverse(sorted_list)
end

function split_nodes_into_sample_observe(g::Graph)
    sample_nodes, observe_nodes = [], []
    for node in g.nodes
        if occursin("sample", node)
            push!(sample_nodes, node)
        elseif occursin("observe", node)
            push!(observe_nodes, node)
        else
            error("Node present that is neither sample nor observe")
        end
    end
    return sample_nodes, observe_nodes
end

### Evaluation ###

function evaluate_node(node, exp, sig, l; fixed_dists=Dict(), fixed_nodes=Dict(), fixed_probs=Dict(), verbose=false)
    if verbose println("Node: ", node) end
    if haskey(fixed_dists, node)
        result = rand(fixed_dists[node])
        p_log_prob = logpdf(eval(exp[2], sig, l), result)
        q_log_prob = logpdf(fixed_dists[node], result)
        sig["logP"] += q_log_prob
        sig["logW"] += p_log_prob - q_log_prob
    elseif haskey(fixed_nodes, node) && haskey(fixed_probs, node)
        result = fixed_nodes[node]
        log_prob = fixed_probs[node]
        sig["logP"] += log_prob
        if occursin("observe", node)
            sig["logW"] += log_prob
        end
    elseif haskey(fixed_nodes, node)
        result = fixed_nodes[node]
        log_prob = logpdf(eval(exp[2], sig, l), result)
        sig["logP"] += log_prob
        if occursin("observe", node)
            sig["logW"] += log_prob
        end
    else
        result = eval(exp, sig, l, verbose=verbose)
    end
    if verbose println("Value: ", result) end
    return result
end

function evaluate_graph(g::Graph; fixed_dists=Dict(), fixed_nodes=Dict(), fixed_probs=Dict(), verbose=false)
    if verbose println(g) end

    sig = Dict("logW" => 0.0, "logP" => 0.0)
    l = Dict{Any, Any}()
    for node in g.nodes
        exp = g.expressions[node]
        original_logP = sig["logP"]
        result = evaluate_node(node, exp, sig, l, fixed_dists=fixed_dists, fixed_nodes=fixed_nodes, fixed_probs=fixed_probs, verbose=verbose)
        l[node] = result
        l[node * "_logP"] = sig["logP"] - original_logP
    end

    result = eval(g.program, sig, l, verbose=verbose)
    if verbose println("Result: ", result) end
    return result, sig, l
end

### Hamiltonian Monte Carlo ###

function generate_IC(g::Graph; verbose=false)
    _, _, l = evaluate_graph(g, verbose=verbose)
    start = [l[node] for node in g.nodes if occursin("sample", node)]
    if verbose println("Initial conditions: ", start) end
    return start
end

function log_joint(g::Graph, x; verbose=false)
    fixed_nodes = Dict()
    i = 1
    for node in g.nodes
        if occursin("sample", node)
            fixed_nodes[node] = x[i]
            i += 1
        end
    end
    _, sig, _ = evaluate_graph(g, fixed_nodes=fixed_nodes, verbose=verbose)
    log_joint = sig["logP"]
    return log_joint
end

### Variational Inference Utilities ###

function save_parameters(parameters::Vector, variationals::Dict)
    params_here = []
    for dist in values(variationals)
        params = [deepcopy(p) for p in dist.params()]
        append!(params_here, params)
    end
    push!(parameters, params_here)
    return parameters
end

function calculate_b(node::String, variational, logQs::Vector, logWs::Vector; zero=false)
    if zero
        b = 0.0
    else
        Fs, Gs = [], []
        for (logQ, logW) in zip(logQs, logWs)
            Q = logQ[node]
            grads = gradient(() -> Q, variational.optim_params())[1]
            G = length(grads) == 1 ? grads[1] : collect(grads)
            for param in variational.optim_params()
                zero!(param.grad)
            end
            F = G * logW
            push!(Fs, F)
            push!(Gs, G)
        end
        Fs = vcat(Fs...)
        Gs = vcat(Gs...)
        cov_FG = sum(covariance(Fs, Gs))
        var_GG = sum(covariance(Gs, Gs))
        b = cov_FG / var_GG
    end
    return b
end

function update_parameters(nodes::Vector{String}, variationals::Dict, logQs::Vector, logWs::Vector, optimizer; zero_b=false)
    total_ELBO, total_loss = 0.0, 0.0
    batch_size = length(logQs)
    for node in nodes
        b = calculate_b(node, variationals[node], logQs, logWs; zero=zero_b)
        ELBO, loss = 0.0, 0.0
        for (logQ, logW) in zip(logQs, logWs)
            ELBO -= logQ[node] * logW
            loss -= logQ[node] * (logW - b)
        end
        ELBO /= batch_size
        loss /= batch_size
        total_ELBO += ELBO
        total_loss += loss
    end
    gradient!(() -> total_loss, optimizer.params())
    step!(optimizer)
    return deepcopy(total_ELBO)
end

function initialize_optimizer(variationals::Dict, learning_rate::Float64)
    all_parameters = []
    for dist in values(variationals)
        parameters = dist.optim_params()
        append!(all_parameters, parameters)
    end
    return Adam(all_parameters, learning_rate)
end

end # module GraphBasedSamplingUtils
