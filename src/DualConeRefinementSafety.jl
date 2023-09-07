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
    lineqs::Vector{Vector{Float64}} # a' * coeffs ≤ 0
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
    @assert all(dfuncs -> length(dfuncs) == length(tmp.funcs), derivs_funcs)
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
    lineqs = Vector{Float64}[]
    for x in points
        @assert length(x) == length(tmp.vars)
        valuation = Dict(zip(tmp.vars, x))
        op_(g) = Symbolics.value(Symbolics.substitute(g, valuation))
        for dfuncs in derivs_funcs
            a = op_.(dfuncs)
            @assert length(a) == nceoff
            push!(lineqs, float.(a))
        end
    end
    return HConeSubset(tmp, lineqs)
end

struct VConeSubset
    tmp::Template
    gens::Vector{Vector{Float64}}
end

function vcone_from_hcone(hc::HConeSubset, lib::Function)
    @assert !isempty(hc.lineqs)
    @assert length(hc.tmp.funcs) > 0
    @assert all(a -> length(a) == length(hc.tmp.funcs), hc.lineqs)
    poly = polyhedron(hrep([HalfSpace(a, 0) for a in hc.lineqs]), lib())
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    @assert all(x -> norm(x) < 1e-6, points(poly))
    gens = collect(float.(r.a) for r in rays(poly))
    @assert all(x -> norm(x) > 1e-6, gens)
    normalize!.(gens)
    return VConeSubset(hc.tmp, gens)
end

struct SoSProblem{GT<:Polynomial,DT<:Polynomial}
    nvert::Int
    gens::Vector{GT}
    derivs::Vector{DT}
end

function sos_problem_from_vcone(vc::VConeSubset, f::Vector{Num})
    tmp = vc.tmp
    op1_(g) = dot(Symbolics.gradient(g, tmp.vars), f)
    op2_(g) = Symbolics.simplify(op1_(g), expand=true)
    funcs = tmp.funcs
    dfuncs = op2_.(funcs)
    vars_dp, = @polyvar x[1:length(tmp.vars)]
    @assert length(tmp.vars) == length(vars_dp)
    valuation = Dict(zip(tmp.vars, vars_dp))
    op_(g) = Symbolics.value(Symbolics.substitute(g, valuation))
    funcs_dp = op_.(funcs)
    dfuncs_dp = op_.(dfuncs)
    gens = [dot(c, funcs_dp) for c in vc.gens]
    derivs = [dot(c, dfuncs_dp) for c in vc.gens]
    @assert length(gens) == length(derivs) == length(vc.gens)
    return SoSProblem(length(vc.gens), gens, derivs)
end

function check_positivity(f, funcs_zero, funcs_pos, solver)
    S = FullSpace()
    for h in funcs_zero
        S = S ∩ (SemialgebraicSets.PolynomialEquality(h))
    end
    for g in funcs_pos
        S = S ∩ (SemialgebraicSets.PolynomialInequality(g))
    end
    model = SOSModel(solver())
    @constraint(model, f ≥ 0, domain=S)
    optimize!(model)
    return primal_status(model) == FEASIBLE_POINT
end

function trim_vertices(sosprob::SoSProblem, solver)
    nvert = sosprob.nvert
    i_bad::Int = 0
    for i = 1:nvert
        f = sosprob.derivs[i]
        funcs_zero = [sosprob.gens[i]]
        funcs_pos = [sosprob.gens[j] for j = 1:nvert if j != i]
        is_good = check_positivity(f, funcs_zero, funcs_pos, solver)
        if !is_good
            i_bad = i
            break
        end
    end
    @assert 0 ≤ i_bad ≤ nvert
    return i_bad
end

# function build_sos_domain(vc::VConeSubset, vars::Vector{Num})
#     @polyvar vars_dp[1:length(vars)]
#     @assert length(vars) == length(vars_dp)
    
#     valuation = Dict(zip(vars, vars_dp))
#     dfuncs = op_(g) = dot(Symbolics.gradient(g, tmp.vars), f) + λ * g
#     for i = 1:maxorder
#         push!(derivs_funcs, [
#             Symbolics.simplify(op_(g), expand=true) for g in derivs_funcs[i]
#         ])
#     end
#     funcs_dp = Symbolics.value.(Symbolics.substitute.(vc.funcs, (valuation,)))
#     set = FullSpace()
#     for gen in vc.gens
#         g_con = dot(gen, funcs_dp)
#         set = set ∩ (SemialgebraicSets.PolynomialInequality(g_con))
#     end
#     return SoSDomain(vars, vars_dp, set)
# end

function g()
    
end

end # module