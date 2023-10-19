module DualConeRefinementSafety

greet() = print("Hello World!")

using LinearAlgebra
using Polyhedra
using DynamicPolynomials
using SumOfSquares

const Template = Vector{<:AbstractPolynomialLike}

struct Field{VT<:Variable,FT<:AbstractPolynomialLike}
    var::Vector{VT}
    flow::Vector{FT}
end

struct Point{VT<:Variable,WT<:Real}
    var::Vector{VT}
    val::Vector{WT}
end

struct HConeSubset{FT<:AbstractPolynomialLike,HT<:HalfSpace}
    funcs::Vector{FT}
    halfspaces::Vector{HT} # h.a' * coeffs ≤ h.β
end

function derivative(g::AbstractPolynomialLike,
                    F::Field,
                    λ::Real)
    return dot(differentiate.(g, F.var), F.flow) + λ * g
end

function derivative(funcs::Template,
                    F::Field,
                    λ::Real)
    return [derivative(g, F, λ) for g in funcs]
end

function hcone_from_points(funcs::Template,
                           F::Field,
                           λ::Real,
                           ϵ::Real,
                           points::Union{AbstractSet,AbstractVector})
    @assert eltype(points) <: Point
    @assert length(funcs) > 0
    dfuncs = derivative(funcs, F, λ)
    halfspaces = HalfSpace{Float64,Vector{Float64}}[]
    for x in points
        @assert length(x.var) == length(x.val)
        a = [g(x.var=>x.val) for g in funcs]
        push!(halfspaces, HalfSpace(a, -ϵ))
        da = [g(x.var=>x.val) for g in dfuncs]
        push!(halfspaces, HalfSpace(da, -ϵ))
    end
    return HConeSubset(funcs, halfspaces)
end

struct VConeSubset{FT<:AbstractPolynomialLike,RT<:Ray}
    funcs::Vector{FT}
    rays::Vector{RT}
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
    @assert length(hc.funcs) > 0
    @assert all(h -> length(h.a) == length(hc.funcs), hc.halfspaces)
    poly = polyhedron(hrep(hc.halfspaces), lib())
    display(points(poly))
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    center = float.(first(points(poly)))
    @assert norm(center) < 1
    rays = collect(float.(r.a) for r in Polyhedra.rays(poly))
    @assert all(r -> norm(r) > 1e-6, rays)
    normalize_shift!.(rays, (center,))
    @assert all(r -> norm(r) ≈ 1, rays)
    return VConeSubset(hc.funcs, [Ray(r) for r in rays])
end

struct SoSConstraint{VT<:AbstractPolynomialLike,
                     DT<:AbstractBasicSemialgebraicSet,
                     ET<:Real}
    val::Vector{VT}
    dom::DT
    ϵ::ET
end

function sos_domain_from_vcone(vc::VConeSubset)
    S = FullSpace()
    for r in vc.rays
        S = S ∩ @set(dot(r.a, vc.funcs) ≤ 0)
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
        @assert length(con.val) == length(c)
        f = dot(c, con.val)
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

function project_generators(rays::Vector{<:Ray},
                            sosprob::SoSProblem,
                            solver;
                            callback_func=(args...) -> nothing)
    @assert !isempty(rays)
    model, c = set_sos_optim(sosprob, solver)
    new_rays = Vector{Ray{Float64,Vector{Float64}}}(undef, length(rays))
    d_max::Float64 = -Inf
    for (i, r) in enumerate(rays)
        c_opt, d = solve_sos_optim(model, c, r.a)
        d_max = max(d, d_max)
        callback_func(i, length(rays), d_max)
        new_rays[i] = Ray(c_opt)
    end
    @assert d_max > -1e-6
    return new_rays, d_max
end

function narrow_vcone!(vc::VConeSubset,
                       dom_init::AbstractBasicSemialgebraicSet,
                       F::Field,
                       λ::Real,
                       ϵ::Real,
                       δ::Real,
                       maxiter,
                       solver;
                       callback_func=(args...) -> nothing)
    ncoeff = length(vc.funcs)
    con_init = SoSConstraint(vc.funcs, dom_init, ϵ)
    dfuncs = derivative(vc.funcs, F, λ)
    iter = 0
    success::Bool = false
    while iter < maxiter && !success
        iter += 1
        dom_deriv = sos_domain_from_vcone(vc)
        con_deriv = SoSConstraint(dfuncs, dom_deriv, ϵ)
        sosprob = SoSProblem(ncoeff, [con_init, con_deriv])
        callback_ = (args...) -> callback_func(iter, args...)
        new_rays, d = project_generators(vc.rays,
                                         sosprob,
                                         solver,
                                         callback_func=callback_)
        if d < δ
            success = true # exit
        else
            for (i, r) in enumerate(new_rays)
                normalize!(r.a)
                vc.rays[i] = r
            end
        end
    end
    return success
end

function find_maximum(f::AbstractPolynomialLike,
                      vc::VConeSubset,
                      solver)
    model = solver()
    r = @variable(model)
    dom = sos_domain_from_vcone(vc)
    @constraint(model, f ≤ r, domain=dom)
    @objective(model, Min, r)
    optimize!(model)
    if primal_status(model) != FEASIBLE_POINT
        display(solution_summary(model))
        error("!FEASIBLE_POINT")
    end
    return objective_value(model)
end

function trim_vcone(vc::VConeSubset, tol::Real, solver)
    remove_set = BitSet()
    for (i, r) in enumerate(vc.rays)
        f = dot(r.a, vc.funcs)
        val_max = find_maximum(f, vc, solver)
        if val_max < -tol
            push!(remove_set, i)
        end
    end
    return remove_set
end

function simplify_vcone!(vc::VConeSubset, tol::Real, solver)
    while true
        remove_set = trim_vcone(vc, tol, solver)
        if !isempty(remove_set)
            deleteat!(vc.rays, remove_set)
        else
            break
        end
    end
    return nothing
end

end # module