module FiniteCone

using LinearAlgebra
using Polyhedra
using DynamicPolynomials

struct Field{VT<:Variable,FT<:AbstractPolynomialLike}
    var::Vector{VT}
    flow::Vector{FT}
end

function derivative(g::AbstractPolynomialLike,
                    F::Field,
                    λ::Real)
    return dot(differentiate.(g, F.var), F.flow) + λ * g
end

elemprod(A, B) = unique(vec(map(x -> x[1] * x[2], Iterators.product(A, B))))

struct FCone{GT<:AbstractPolynomialLike}
    gens::Vector{GT}
end

function init_cone(maxdeg::Int,
                   gens_init::Vector{<:AbstractPolynomialLike},
                   gens_pos::Vector{<:AbstractPolynomialLike},
                   lib)
    crossgens = elemprod(gens_init, gens_pos)
    lmul!(-1, crossgens)
    var = variables(crossgens)
    base = monomials(var, 0:maxdeg)
    gens = vcat(base, crossgens)
    d = maximum(p -> maxdegree(p), gens, init=0)::Int
    @assert d ≥ 0
    monos = monomials(var, 0:d)
    nm = length(monos)
    nb = length(base)
    ng = length(gens)
    nrow = nm + ng - nb
    T = coefficient_type(eltype(gens))
    A = zeros(T, nrow, ng)
    for (i, g) in enumerate(gens)
        k = nrow * (i - 1) + 1
        copyto!(A, k, coefficients(g, monos))
    end
    for i = 1:(ng - nb)
        A[nm + i, nb + i] = -1
    end
    b = zeros(T, nrow)
    hp = polyhedron(hrep(A, b, BitSet(1:nm)), lib())
    poly = project(hp, 1:nb)
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    @assert all(iszero, points(poly))
    rays = collect(r.a for r in Polyhedra.rays(poly))
    @assert all(r -> !iszero(r), rays)
    @assert all(r -> length(r) == nb, rays)
    return FCone([dot(r, base) for r in rays])
end

function refine_cone(fc::FCone,
                     F::Field,
                     λ::Real,
                     gens_pos::Vector{<:AbstractPolynomialLike},
                     lib)
    crossgens = elemprod(fc.gens, gens_pos)
    lmul!(-1, crossgens)
    dgens = [derivative(g, F, λ) for g in fc.gens]
    gens = vcat(dgens, crossgens)
    var = variables(gens)
    d = maximum(p -> maxdegree(p), gens, init=0)::Int
    @assert d ≥ 0
    monos = monomials(var, 0:d)
    nm = length(monos)
    nd = length(dgens)
    ng = length(gens)
    nrow = nm + ng
    T = coefficient_type(eltype(gens))
    A = zeros(T, nrow, ng)
    for (i, g) in enumerate(gens)
        k = nrow * (i - 1) + 1
        copyto!(A, k, coefficients(g, monos))
    end
    for i = 1:ng
        A[nm + i, i] = -1
    end
    b = zeros(T, nrow)
    hp = polyhedron(hrep(A, b, BitSet(1:nm)), lib())
    poly = project(hp, 1:nd)
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    @assert all(iszero, points(poly))
    rays = collect(r.a for r in Polyhedra.rays(poly))
    @assert all(r -> !iszero(r), rays)
    @assert all(r -> length(r) == nd, rays)
    return FCone([dot(r, fc.gens) for r in rays])
end

end