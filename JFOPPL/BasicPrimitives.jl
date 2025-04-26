module BasicPrimitives

using Random
using LinearAlgebra
using .DistributionsPrep: distribution_constructors

vector(xs...) = collect(xs)

function _get(container, idx)
    if container isa AbstractArray
        return container[Int(idx)+1]
    elseif container isa Dict
        return container[idx]
    else
        error("`get` not defined for $(typeof(container))")
    end
end

function _put(container, idx, val)
    if container isa AbstractArray
        container[Int(idx)+1] = val
        return container
    elseif container isa Dict
        container[idx] = val
        return container
    else
        error("`put` not defined for $(typeof(container))")
    end
end

_append(a::AbstractVector, v) = (push!(a, v); a)
hashmap(kv...) = Dict(kv...)
_and(a,b) = a && b
_or(a,b)  = a || b
_mat_tanh(M) = tanh.(M)

# Convert a nested vector to a proper Matrix{Float64}
function to_matrix(V)
    V isa AbstractMatrix && return V
    rows = length(V)
    cols = length(V[1])
    M = Array{Float64}(undef, rows, cols)
    @inbounds for i in 1:rows, j in 1:cols
        M[i,j] = Float64(V[i][j])
    end
    return M
end

# Create a zero-based categorical distribution
function zero_based_categorical(p::AbstractVector{<:Real})
    cats = collect(0:length(p)-1)    # categories 0,1,2,...
    return DiscreteNonParametric(cats, p)
end


const primitives = Dict{String,Any}(

    # logic & comparison
    "<"   => (<),
    "<="  => (<=),
    ">"   => (>),
    ">="  => (>=),
    "="   => (==),
    "and" => _and,
    "or"  => _or,

    # arithmetic
    "+"   => (+),
    "-"   => (-),
    "*"   => (*),
    "/"   => (/),
    "exp" => exp,
    "sqrt"=> sqrt,
    "abs" => abs,

    # container ops
    "vector" => vector,
    "get"    => _get,
    "put"    => _put,
    "append" => _append,
    "first"  => xs -> xs[1],
    "second" => xs -> xs[2],
    "last"   => xs -> xs[end],
    "rest"   => xs -> xs[2:end],
    "hash-map" => hashmap,

    # matrix ops
    "mat-mul"       => (A,B)   -> to_matrix(A) * to_matrix(B),
    "mat-add"       => (A,B)   -> to_matrix(A) .+ to_matrix(B),
    "mat-transpose" => M       -> transpose(to_matrix(M)),
    "mat-repmat"    => (M,r,c) -> repeat(to_matrix(M), (Int(r),Int(c))),
    "mat-tanh"      => _mat_tanh,

    # “normal” primitive
    "normal"        => (μ, σ; kwargs...) -> Normal(float(μ), float(σ)),

    # discrete / categorical
    "discrete"       => (p; kwargs...) -> zero_based_categorical(p),
    "discrete-guide" => (p; kwargs...) -> zero_based_categorical(p),

    # all distribution constructors and their guides
    distribution_constructors...,
    [k * "-guide" => v for (k,v) in distribution_constructors]...,

    # domain-specific stub
    "oneplanet" => (args...)->error("`oneplanet` not implemented in Julia backend.")
)

export primitives

end # module
