module DualConeRefinementSafety

greet() = print("Hello World!")

using LinearAlgebra
using Symbolics
using Polyhedra
using DynamicPolynomials
using SumOfSquares

struct Template
    vars::Vector{Num}
    funcs::Vector{Num}
end

struct HConeSubset
    tmp::Template
    supps::Vector{Vector{Float64}} # a' * coeffs ≤ 0
end

function _shifted_derivatives(tmp::Template,
                              f::Vector{Num},
                              λ::Real,
                              maxorder::Int)
    derivs_funcs::Vector{Vector{Num}} = [tmp.funcs]
    op1_(g) = dot(Symbolics.gradient(g, tmp.vars), f) + λ * g
    op2_(g) = Symbolics.simplify(op1_(g), expand=true)
    for i = 1:maxorder
        push!(derivs_funcs, op2_.(derivs_funcs[i]))
    end
    @assert length(derivs_funcs) == maxorder + 1
    @assert all(duncs -> length(duncs) == length(tmp.funcs), derivs_funcs)
    return derivs_funcs
end

function hcone_from_points(tmp::Template,
                           f::Vector{Num},
                           λ::Real,
                           maxorder::Int,
                           points::Vector{<:AbstractVector{<:Real}})
    nceoff = length(tmp.funcs)
    @assert nceoff > 0
    derivs_funcs = _shifted_derivatives(tmp, f, λ, maxorder)
    supps = Vector{Float64}[]
    for x in points
        @assert length(x) == length(tmp.vars)
        valuation = Dict(zip(tmp.vars, x))
        op_(g) = Symbolics.value(Symbolics.substitute(g, valuation))
        for duncs in derivs_funcs
            a = op_.(duncs)
            @assert length(a) == nceoff
            push!(supps, float.(a))
        end
    end
    return HConeSubset(tmp, supps)
end

struct VConeSubset
    tmp::Template
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

struct SoSProblem{GT<:Polynomial,DT<:Polynomial}
    indices::BitSet
    doms::Vector{GT}
    vals::Vector{DT}
end

function sos_problem_from_vcone(vc::VConeSubset, f::Vector{Num})
    tmp = vc.tmp
    op1_(g) = dot(Symbolics.gradient(g, tmp.vars), f)
    op2_(g) = Symbolics.simplify(op1_(g), expand=true)
    funcs = tmp.funcs
    duncs = op2_.(funcs)
    vars_dp, = @polyvar x[1:length(tmp.vars)]
    T = typeof(1.0 + sum(vars_dp))
    @assert length(tmp.vars) == length(vars_dp)
    valuation = Dict(zip(tmp.vars, vars_dp))
    op_(g)::T = Symbolics.value(Symbolics.substitute(g, valuation)) + zero(T)
    funcs_dp = op_.(funcs)
    duncs_dp = op_.(duncs)
    doms = [dot(c, funcs_dp)::T for c in vc.gens]
    vals = [dot(c, duncs_dp)::T for c in vc.gens]
    nvert = length(vc.gens)
    @assert length(doms) == length(vals) == nvert
    return SoSProblem(BitSet(1:nvert), doms, vals)
end

function check_positivity(f::Polynomial,
                          hs::Vector{<:Polynomial},
                          indices::BitSet,
                          eqset::BitSet,
                          solver)
    S = FullSpace()
    for i in indices
        if i in eqset
            S = S ∩ (SemialgebraicSets.PolynomialEquality(hs[i]))
        else
            S = S ∩ (SemialgebraicSets.PolynomialInequality(hs[i]))
        end
    end
    model = SOSModel(solver())
    @constraint(model, f ≥ 0, domain=S)
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

function narrow_vcone(vc::VConeSubset, f::Vector{Num},
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