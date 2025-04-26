module BasicPrimitives

using Random
using LinearAlgebra
using .DistributionsPrep: distribution_constructors

# helpers
vector(xs...) = collect(xs)

function _get(container, idx)
    container isa AbstractArray && return container[Int(idx)+1]
    container isa Dict          && return container[idx]
    error("`get` not defined for $(typeof(container))")
end

function _put(container, idx, val)
    container isa AbstractArray && (container[Int(idx)+1] = val;  return container)
    container isa Dict          && (container[idx] = val;         return container)
    error("`put` not defined for $(typeof(container))")
end

_append(a::AbstractVector, v) = (push!(a, v); a)

hashmap(kv...) = Dict(kv...)

# logic / maths
_and(a,b) = a && b
_or(a,b)  = a || b

# matrix helpers
_mat_transpose(M)  = transpose(M)
_mat_tanh(M)       = tanh.(M)
_mat_repmat(M,r,c) = repeat(M,(Int(r),Int(c)))

# ❸  build the master dictionary
const primitives = Dict{String,Any}(

    # logic & comparison
    "<" => (<), "<=" => (<=), ">" => (>), ">=" => (>=), "=" => (==),
    "and" => _and, "or" => _or,

    # maths
    "+" => (+), "-" => (-), "*" => (*), "/" => (/),
    "exp" => exp, "sqrt" => sqrt, "abs" => abs,

    # containers
    "vector" => vector, "get" => _get, "put" => _put, "append" => _append,
    "first" => xs->xs[1][1], "second" => xs->xs[1][2],
    "last" => xs->xs[1][end], "rest" => x->x[2:end],
    "hash-map" => hashmap,

    # matrix ops
    "mat-transpose" => _mat_transpose,  "mat-add" => (+),
    "mat-mul" => (*), "mat-tanh" => _mat_tanh, "mat-repmat" => _mat_repmat,

    # all distribution constructors
    distribution_constructors...,

    # “guide” aliases
    [k*"-guide"=>v for (k,v) in distribution_constructors]...,

    # domain-specific
    "oneplanet" => (args...)->error("`oneplanet` not implemented in Julia backend.")
)

export primitives
end     # module
