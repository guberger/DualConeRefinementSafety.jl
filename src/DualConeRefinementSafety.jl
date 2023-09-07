module DualConeRefinementSafety

greet() = print("Hello World!")

using LinearAlgebra
using Polyhedra
using DynamicPolynomials
using SumOfSquares

struct Template{VT<:Variable,FT<:AbstractPolynomialLike}
    vars::Vector{VT}
    funcs::Vector{FT}
end

struct HConeSubset{TT<:Template}
    tmp::TT
    supps::Vector{Vector{Float64}} # a' * coeffs ≤ 0
end

function _shifted_derivatives(tmp::Template,
                              f::Vector{<:AbstractPolynomialLike},
                              λ::Real,
                              maxorder::Int)
    T = promote_type(eltype(tmp.funcs), eltype(f), typeof(λ))
    derivs_funcs::Vector{Vector{T}} = [convert.(T, tmp.funcs)]
    op_(g)::T = convert(T, dot(differentiate.(g, tmp.vars), f) + λ * g)
    for i = 1:maxorder
        push!(derivs_funcs, op_.(derivs_funcs[i]))
    end
    @assert length(derivs_funcs) == maxorder + 1
    @assert all(duncs -> length(duncs) == length(tmp.funcs), derivs_funcs)
    return derivs_funcs
end

function hcone_from_points(tmp::Template,
                           f::Vector{<:AbstractPolynomialLike},
                           λ::Real,
                           maxorder::Int,
                           points::Union{AbstractSet,AbstractVector})
    nceoff = length(tmp.funcs)
    @assert nceoff > 0
    derivs_funcs = _shifted_derivatives(tmp, f, λ, maxorder)
    supps = Vector{Float64}[]
    for x in points
        @assert eltype(x) <: Real
        @assert length(x) == length(tmp.vars)
        op_(g)::Float64 = Float64(g(tmp.vars=>x))
        for duncs in derivs_funcs
            a = op_.(duncs)
            @assert length(a) == nceoff
            push!(supps, a)
        end
    end
    return HConeSubset(tmp, supps)
end

struct VConeSubset{TT<:Template}
    tmp::TT
    gens::Vector{Vector{Float64}}
end

function vcone_from_hcone(hc::HConeSubset, lib::Function)
    @assert !isempty(hc.supps)
    @assert length(hc.tmp.funcs) > 0
    @assert all(a -> length(a) == length(hc.tmp.funcs), hc.supps)
    poly = polyhedron(hrep([HalfSpace(a, 0) for a in hc.supps]), lib())
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    @assert all(x -> norm(x) < 1e-6, points(poly))
    gens = collect(float.(r.a) for r in rays(poly))
    @assert all(x -> norm(x) > 1e-6, gens)
    normalize!.(gens)
    return VConeSubset(hc.tmp, gens)
end

struct SoSProblem{GT<:AbstractPolynomialLike,DT<:AbstractPolynomialLike}
    indices::BitSet
    doms::Vector{GT}
    vals::Vector{DT}
end

function sos_problem_from_vcone(vc::VConeSubset, f::Vector{<:AbstractPolynomialLike})
    tmp = vc.tmp
    T = promote_type(eltype(tmp.funcs), eltype(f), Float64)
    op_(g)::T = convert(T, dot(differentiate.(g, tmp.vars), f))
    funcs = convert.(T, tmp.funcs)
    duncs = op_.(funcs)
    doms = [dot(c, funcs)::T for c in vc.gens]
    vals = [dot(c, duncs)::T for c in vc.gens]
    nvert = length(vc.gens)
    @assert length(doms) == length(vals) == nvert
    return SoSProblem(BitSet(1:nvert), doms, vals)
end

function check_positivity(f::AbstractPolynomialLike,
                          hs::Vector{<:AbstractPolynomialLike},
                          indices::BitSet,
                          eqset::BitSet,
                          solver)
    S = FullSpace()
    for i in indices
        S = i ∈ eqset ? S ∩ @set(hs[i] == 0) : S ∩ @set(hs[i] ≤ 0)
    end
    model = SOSModel(solver())
    @constraint(model, f ≤ 0, domain=S)
    optimize!(model)
    return primal_status(model) == FEASIBLE_POINT
end

function trim_vertices(sosprob::SoSProblem, solver;
                       callback_func=(args...) -> nothing)
    indices = sosprob.indices
    @assert !isempty(indices)
    hs = sosprob.doms
    eqset = BitSet()
    i_bad::Int = 0
    is_good = false
    for (iter, i) in enumerate(indices)
        empty!(eqset); push!(eqset, i)
        f = sosprob.vals[i]
        is_good = check_positivity(f, hs, indices, eqset, solver)
        callback_func(iter, i, indices, is_good)
        if !is_good
            i_bad = i
            break
        end
    end
    @assert i_bad == 0 || (is_good && i_bad ∈ indices)
    return i_bad
end

function narrow_vcone(vc::VConeSubset, f::Vector{<:AbstractPolynomialLike},
                      maxiter, solver;
                      callback_func=(args...) -> nothing)
    sosprob = sos_problem_from_vcone(vc, f)
    iter = 0
    success::Bool = false
    while iter < maxiter && !success
        iter += 1
        callback_ = (args...) -> callback_func(iter, args...)
        i_bad = trim_vertices(sosprob, solver, callback_func=callback_)
        success = (i_bad == 0)
        delete!(sosprob.indices, i_bad)
    end
    return VConeSubset(vc.tmp, [vc.gens[i] for i in sosprob.indices]), success
end

end # module