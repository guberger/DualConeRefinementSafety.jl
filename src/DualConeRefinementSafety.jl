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
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    center = float.(first(points(poly)))
    @assert norm(center) < 1
    rays = collect(float.(r.a) for r in Polyhedra.rays(poly))
    @assert all(r -> norm(r) > 1e-6, rays)
    normalize_shift!.(rays, (center,))
    @assert all(c -> norm(c) ≈ 1, rays)
    return VConeSubset(hc.funcs, [Ray(c) for c in rays])
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

crossprod(A, B) = unique(vec(map(x -> x[1] * x[2], Iterators.product(A, B))))

const PT = DynamicPolynomials.Commutative{DynamicPolynomials.CreationOrder}
const MT = Monomial{PT,Graded{LexOrder}}

function _monomials(ps::Union{AbstractSet,AbstractVector})
    @assert eltype(ps) <: AbstractPolynomialLike
    monos = Set{MT}()
    for p in ps
        for mono in monomials(p)
            push!(monos, mono)
        end
    end
    return monos    
end

function closure_cone(funcs::Vector{<:AbstractPolynomialLike},
                      gens_init::Vector{<:AbstractPolynomialLike},
                      ϵ::Real,
                      gens_sos::Vector{<:AbstractPolynomialLike},
                      lib)
    crossgens = crossprod(gens_init, gens_sos)
    lmul!(-1, crossgens)
    gens = vcat(funcs, crossgens, [Monomial(1)])
    monos = collect(_monomials(gens))
    nm = length(monos)
    nf = length(funcs)
    ng = length(gens)
    nrow, ncol = nm + ng, ng
    A = zeros(nrow, ncol)
    for (i, g) in enumerate(gens)
        io = nrow * (i - 1) + 1
        copyto!(A, io, coefficients(g, monos))
    end
    for i = (nf + 1):ng
        A[nm + i, i] = -1
    end
    b = zeros(nrow)
    b[nm + ng] = -ϵ
    hp = polyhedron(hrep(A, b, BitSet(1:nm)), lib())
    poly = project(hp, 1:nf)
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    center = float.(first(points(poly)))
    @assert norm(center) < 1
    rays = collect(float.(r.a) for r in Polyhedra.rays(poly))
    @assert all(r -> norm(r) > 1e-6, rays)
    normalize_shift!.(rays, (center,))
    @assert all(c -> norm(c) ≈ 1, rays)
    return VConeSubset(funcs, [Ray(c) for c in rays])
end

function refine_cone(vc::VConeSubset,
                     F::Field,
                     λ::Real,
                     ϵ::Real,
                     gens_sos::Vector{<:AbstractPolynomialLike},
                     lib)
    cone_gens = [dot(r.a, vc.funcs) for r in vc.rays]
    crossgens = crossprod(cone_gens, gens_sos)
    lmul!(-1, crossgens)
    dfuncs = derivative(vc.funcs, F, λ)
    gens = vcat(dfuncs, crossgens, [Monomial(1)])
    monos = collect(_monomials(gens))
    nm = length(monos)
    nf = length(vc.funcs)
    ng = length(gens)
    nr = length(vc.rays)
    nrow, ncol = nm + nf + ng + nr, nf + ng + nr
    A = zeros(nrow, ncol)
    for (i, g) in enumerate(gens)
        io = nrow * (i - 1) + 1
        copyto!(A, io, coefficients(g, monos))
    end
    for (i, r) in enumerate(vc.rays)
        io = nrow * (nf + ng + i - 1) + nm + 1
        copyto!(A, io, r.a)
    end
    for i = 1:ncol
        A[nm + i, i] = -1
    end
    b = zeros(nrow)
    b[nm + nf + ng] = -ϵ
    hp = polyhedron(hrep(A, b, BitSet(1:(nm + nf))), lib())
    poly = project(hp, 1:nf)
    @assert isempty(lines(poly))
    @assert length(points(poly)) == 1
    center = float.(first(points(poly)))
    @assert norm(center) < 1
    rays = collect(float.(r.a) for r in Polyhedra.rays(poly))
    @assert all(r -> norm(r) > 1e-6, rays)
    normalize_shift!.(rays, (center,))
    @assert all(c -> norm(c) ≈ 1, rays)
    return VConeSubset(vc.funcs, [Ray(c) for c in rays])
end

# function refine_cone(fcone::FCone,
#                      F::Field,
#                      λ::Real,
#                      fcone_sos::FCone,
#                      lib)
#     D(g) = dot(differentiate.(g, F.var), F.flow) + λ * g
#     dgens = D.(fcone.generators)
#     cgens = crossprod(fcone.generators, fcone_sos.generators)
#     lmul!(-1, cgens)
#     gens = vcat(dgens, cgens)
#     monos = collect(_monomials(gens))

#     nm = length(monos)
#     ng = length(gens)
#     A = zeros(nm + ng, ng)
#     for (i, g) in enumerate(gens)
#         Ao = (nm + ng) * (i - 1) + 1
#         copyto!(A, Ao, coefficients(g, monos))
#     end
#     for i = 1:ng
#         A[nm + i, i] = -1
#     end
#     b = zeros(nm + ng)
#     hp = polyhedron(hrep(A, b, BitSet(1:nm)), lib())
#     nc = length(fcone.generators)
#     poly = project(hp, 1:nc)
#     @assert length(points(poly)) == 1
#     @assert all(x -> norm(x) < 1e-6, points(poly))
#     verts = [r.a for r in rays(poly)]
#     @assert all(x -> norm(x) > 1e-6, verts)
#     @assert all(x -> length(x) == nc, verts)
#     normalize!.(verts)
#     nondec = true
#     c = zeros(Int, nc)
#     for i = 1:nc
#         c[i] = 1
#         if c ∉ poly
#             nondec = false
#             break
#         end
#         c[i] = 0
#     end
#     return FCone([dot(c, fcone.generators) for c in verts]), nondec
# end

end # module