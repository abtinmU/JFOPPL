module EvaluationBasedSampling

using Random
using Statistics
using LinearAlgebra
using .BasicPrimitives: primitives, distributions
using .Distributions

export AbstractSyntaxTree, eval, bind_functions, evaluate_program

# Define the AbstractSyntaxTree type
struct AbstractSyntaxTree
    functions::Vector{Any}
    program::Any

    function AbstractSyntaxTree(ast_json::Vector{Any})
        new(ast_json[1:end-1], ast_json[end])
    end
end

# Evaluate function
function eval(e, sig, l, rho=Dict{String,Any}(); verbose=false)
    if verbose println("Expression (before): ", e) end

    if e isa Number || e isa Bool
        result = e

    elseif e isa String
        result = l[e]

    elseif e isa Array
        if e[1] == "defn"
            error("This defn case should never happen!")

        elseif e[1] == "let"
            expression, name = e[2][2], e[2][1]
            c1 = eval(expression, sig, l, rho)
            l[name] = c1
            result = eval(e[3], sig, l, rho)

        elseif e[1] == "if"
            e1 = eval(e[2], sig, l, rho)
            result = e1 ? eval(e[3], sig, l, rho) : eval(e[4], sig, l, rho)

        elseif e[1] in ["sample", "sample*"]
            d = eval(e[2], sig, l, rho)
            s = rand(d)
            log_prob = logpdf(d, s)
            sig["logP"] += log_prob
            result = s

        elseif e[1] in ["observe", "observe*"]
            d = eval(e[2], sig, l, rho)
            y = eval(e[3], sig, l, rho)
            log_prob = logpdf(d, y)
            sig["logP"] += log_prob
            sig["logW"] += log_prob
            result = y

        else
            cs = [eval(element, sig, l, rho) for element in e[2:end]]

            if e[1] isa Array
                println("List: ", e[1])
                error("This list case should never happen!")

            elseif (e[1] isa String) && (haskey(rho, e[1]))
                variables, function_body = rho[e[1]]
                func_env = deepcopy(l)
                for (variable, exp) in zip(variables, cs)
                    func_env[variable] = exp
                end
                func_env[e[1]] = function_body
                result = eval(function_body, sig, func_env, rho)

            elseif (e[1] isa String) && (e[1] in distributions) && (haskey(primitives, e[1]))
                result = primitives[e[1]](cs...; validate_args=false)

            elseif (e[1] isa String) && (haskey(primitives, e[1]))
                result = primitives[e[1]](cs...)

            else
                println("List expression not recognised: ", e)
                error("List expression not recognised")
            end
        end

    else
        println("Expression not recognised: ", e)
        error("Expression not recognised")
    end

    if verbose
        println("Expression (after): ", e)
        println("Result: ", result, typeof(result))
    end

    return result
end

# Bind functions
function bind_functions(ast::AbstractSyntaxTree)
    rho = Dict{String,Any}()
    for e in ast.functions
        if e[1] == "defn"
            rho[e[2]] = (e[3], e[4])
        end
    end
    return rho
end

# Evaluate program
function evaluate_program(ast::AbstractSyntaxTree; verbose=false)
    sig = Dict("logW" => 0.0, "logP" => 0.0)
    l = Dict{String,Any}()
    rho = bind_functions(ast)
    e = eval(ast.program, sig, l, rho; verbose=verbose)
    return e, sig, l
end

end # module EvaluationBasedSampling