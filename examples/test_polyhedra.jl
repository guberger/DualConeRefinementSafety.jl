using LinearAlgebra
using Polyhedra
using CDDLib
using DynamicPolynomials

cprod(A, B) = unique(vec(map(x -> x[1] * x[2], Iterators.product(A, B))))
max_degree(As...) = maximum(A -> maximum(x -> maxdegree(x), A, init=0), As)

vars, = @polyvar x[1:2]
f = [
    x[2] / 2 + x[1] - x[1] * (4 * x[1]^2 + x[2]^2),
    -2 * x[1] + x[2] - x[2] * (4 * x[1]^2 + x[2]^2),
]
display(f)
rad = 0.5
funcs_init = [+x[1] - rad, +x[2] - rad, -x[1] - rad, -x[2] - rad]

sos_monos = monomials(vars, 0:1)
sos_duos = filter(e -> e[1] != e[2],
                  vec(collect(Iterators.product(sos_monos, sos_monos))))
sos_gens = filter(p -> !iszero(p), unique(vcat(
    vec(map(e -> e^2, sos_monos)),
    vec(map(e -> (e[1] + e[2])^2, sos_duos)),
    vec(map(e -> (e[1] - e[2])^2, sos_duos)),
)))
display(sos_gens)
λ = 1

function refine_cone(S::Vector{<:AbstractPolynomialLike},
                     vars::AbstractVector{<:Variable},
                     f::AbstractVector{<:AbstractPolynomialLike},
                     λ::Real,
                     sos_gens::Union{AbstractSet,AbstractVector},
                     lib)
    D(g) = dot(differentiate.(g, vars), f) + λ * g
    @assert eltype(sos_gens) <: AbstractPolynomialLike
    DS = D.(S)
    CS = cprod(S, sos_gens)
    d = max_degree(S, DS, CS)
    @assert isinteger(d) && d ≥ 0
    monos = vec(monomials(vars, 0:d))
    nm = length(monos)

    nλ = length(DS) # λ
    nμ = length(CS) # μ
    A1 = Matrix{Float64}(undef, nm, nλ + nμ)
    for (i, g) in enumerate(DS)
        A1[:, i] = +coefficients(g, monos)
    end
    for (i, g) in enumerate(CS)
        A1[:, nλ + i] = -coefficients(g, monos)
    end
    A2 = -Matrix{Float64}(I, nλ + nμ, nλ + nμ)
    A = vcat(A1, A2)
    b = zeros(nm + nλ + nμ)
    hp = polyhedron(hrep(A, b, BitSet(1:nm)), lib())
    p = project(hp, 1:nλ)
    @assert length(points(p)) == 1
    @assert all(x -> norm(x) < 1e-6, points(p))
    verts = [r.a for r in rays(p)]
    @assert all(x -> norm(x) > 1e-6, verts)
    @assert all(x -> length(x) == nλ, verts)
    normalize!.(verts)
    nondec = true
    for i = 1:nλ
        c = [j == i ? 1 : 0 for j = 1:nλ]
        nondec &= c ∈ p
        nondec || break
    end
    return [dot(c, S) for c in verts], nondec
end

S = copy(funcs_init)
for i = 1:10
    global S
    display(i)
    S, stable = refine_cone(S, vars, f, λ, sos_gens, () -> CDDLib.Library())
    stable && break
end
display(S)

using Plots

plt = plot(xlabel="x1", ylabel="x2", aspect_ratio=:equal)

x1s_ = range(-2, 2, length=10)
x2s_ = range(-4, 4, length=10)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[fi(vars=>x) for fi in f] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.4 / nx
dxs2 = getindex.(dxs, 2) * 0.4 / nx
quiver!(x1s, x2s, quiver=(dxs1, dxs2))

x1s_ = range(-2, 2, length=100)
x2s_ = range(-4, 4, length=100)
Fplot_init(x1, x2) = maximum(g(vars=>[x1, x2]) for g in S)
z = @. Fplot_init(x1s_', x2s_)
contour!(x1s_, x2s_, z, levels=[0])
display(current())