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

struct HConeSubset{TT<:Template,HT<:HalfSpace}
    tmp::TT
    halfspaces::Vector{HT} # h.a' * coeffs ≤ h.β
end

function deriv_template(tmp::Template,
                        f::Vector{<:AbstractPolynomialLike},
                        λ::Real)
    op_(g) = dot(differentiate.(g, tmp.vars), f) + λ * g
    return [op_(g) for g in tmp.funcs]
end

function hcone_from_points(tmp::Template,
                           f::Vector{<:AbstractPolynomialLike},
                           λ::Real,
                           ϵ::Real,
                           points::Union{AbstractSet,AbstractVector})
    @assert length(tmp.funcs) > 0
    dfuncs = deriv_template(tmp, f, λ)
    halfspaces = HalfSpace{Float64,Vector{Float64}}[]
    for x in points
        @assert x isa AbstractVector{<:Real}
        @assert length(x) == length(tmp.vars)
        a = [g(tmp.vars=>x) for g in tmp.funcs]
        push!(halfspaces, HalfSpace(a, -ϵ))
        a = [g(tmp.vars=>x) for g in dfuncs]
        push!(halfspaces, HalfSpace(a, -ϵ))
    end
    return HConeSubset(tmp, halfspaces)
end

struct VConeSubset{TT<:Template}
    tmp::TT
    vertices::Vector{Vector{Float64}}
end

# Find λ > 0 such that norm(c + λ*v) = 1
# I.e., norm(v)^2 * λ^2 + 2 * dot(c, v) * λ + norm(c)^2 = 1
function normalize_shift!(v, c)
    a1, a2, a3 = norm(v)^2, 2 * dot(v, c), norm(c)^2 - 1
    Δ = a2^2 - 4 * a1 * a3
    @assert Δ > a2^2
    λ = (-a2 + sqrt(Δ)) / (2 * a1)
    @assert λ > 0
    map!((vi, ci) -> λ * vi + ci, v, v, c)
end

function vcone_from_hcone(hc::HConeSubset, lib::Function)
    @assert !isempty(hc.halfspaces)
    @assert length(hc.tmp.funcs) > 0
    @assert all(h -> length(h.a) == length(hc.tmp.funcs), hc.halfspaces)
    poly = polyhedron(hrep(hc.halfspaces), lib())
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    center = float.(first(points(poly)))
    @assert norm(center) < 1
    vertices = collect(float.(r.a) for r in rays(poly))
    @assert all(r -> norm(r) > 1e-6, vertices)
    normalize_shift!.(vertices, (center,))
    @assert all(c -> norm(c) ≈ 1, vertices)
    return VConeSubset(hc.tmp, vertices)
end

struct SoSConstraint{VT<:AbstractPolynomialLike,
                     DT<:AbstractBasicSemialgebraicSet,
                     ET<:Real}
    vals::Vector{VT}
    dom::DT
    ϵ::ET
end

function sos_domain_from_vcone(vc::VConeSubset)
    S = FullSpace()
    for c in vc.vertices
        S = S ∩ @set(dot(c, vc.tmp.funcs) ≤ 0)
    end
    return S
end

function sos_domain_from_funcs(funcs::Vector{<:AbstractPolynomialLike})
    S = FullSpace()
    for g in funcs
        S = S ∩ @set(g ≤ 0)
    end
    return S
end

struct SoSProblem{CT<:SoSConstraint}
    ncoeff::Int
    cons::Vector{CT}
end

function set_sos_optim(sosprob::SoSProblem, solver)
    model = solver()
    c = @variable(model, [1:sosprob.ncoeff])
    @constraint(model, dot(c, c) ≤ 1)
    for con in sosprob.cons
        @assert length(con.vals) == length(c)
        f = dot(c, con.vals)
        @constraint(model, f ≤ -con.ϵ, domain=con.dom)
    end
    return model, c
end

function solve_sos_optim(model::Model,
                         c::AbstractVector{<:VariableRef},
                         c0::AbstractVector{<:Real})
    @assert length(c0) == length(c)
    set_objective_sense(model, MOI.FEASIBILITY_SENSE) # workaround ...
    @objective(model, Min, dot(c - c0, c - c0))
    optimize!(model)
    if primal_status(model) != FEASIBLE_POINT
        display(solution_summary(model))
        error("!FEASIBLE_POINT")
    end
    return value.(c), objective_value(model)
end

function project_generators(vertices::Vector{<:AbstractVector{<:Real}},
                            sosprob::SoSProblem,
                            solver;
                            callback_func=(args...) -> nothing)
    @assert !isempty(vertices)
    model, c = set_sos_optim(sosprob, solver)
    new_vertices = Vector{Vector{Float64}}(undef, length(vertices))
    r_max::Float64 = -Inf
    for (i, c0) in enumerate(vertices)
        c_opt, r = solve_sos_optim(model, c, c0)
        r_max = max(r, r_max)
        callback_func(i, length(vertices), r_max)
        new_vertices[i] = c_opt
    end
    @assert r_max > -1e-6
    return new_vertices, r_max
end

function narrow_vcone!(vc::VConeSubset,
                       funcs_init::Vector{<:AbstractPolynomialLike},
                       f::Vector{<:AbstractPolynomialLike},
                       λ::Real,
                       ϵ::Real,
                       δ::Real,
                       maxiter,
                       solver;
                       callback_func=(args...) -> nothing)
    ncoeff = length(vc.tmp.funcs)
    dom_init = sos_domain_from_funcs(funcs_init)
    con_init = SoSConstraint(vc.tmp.funcs, dom_init, ϵ)
    dfuncs = deriv_template(vc.tmp, f, λ)
    iter = 0
    success::Bool = false
    while iter < maxiter && !success
        iter += 1
        dom_deriv = sos_domain_from_vcone(vc)
        con_deriv = SoSConstraint(dfuncs, dom_deriv, ϵ)
        sosprob = SoSProblem(ncoeff, [con_init, con_deriv])
        callback_ = (args...) -> callback_func(iter, args...)
        new_vertices, r = project_generators(vc.vertices,
                                             sosprob,
                                             solver,
                                             callback_func=callback_)
        if r < δ
            success = true # exit
        else
            for (i, c) in enumerate(new_vertices)
                vc.vertices[i] = normalize!(c)
            end
        end
    end
    return success
end

end # module