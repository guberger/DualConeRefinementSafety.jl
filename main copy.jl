module SyntheticExample

using LinearAlgebra
using Symbolics
# Two options for computing vertices of a polyhedron
# Choose between
# - "CDDLib" (via Polyhedra.jl https://github.com/JuliaPolyhedra/Polyhedra.jl)
# - "QHull" (via SciPy https://docs.scipy.org/doc/scipy/reference/spatial.html)
const VERT_ENUM_METH = "CDDLib"
@static if VERT_ENUM_METH == "CDDLib"
    using Polyhedra
    using CDDLib
end
@static if VERT_ENUM_METH == "QHull"
    using PyCall
    const spatial = pyimport_conda("scipy.spatial", "scipy")
end
using DynamicPolynomials
using SumOfSquares
using MosekTools
using Plots

# Create the variables for symbolic manipulation
Symbolics.@variables x1, x2
x = [x1, x2]

# Just some helper functions
subs_ = Symbolics.substitute
eval_(val1, val2) = Dict([x1 => val1, x2 => val2])
eval_(f::Num, valuation::Dict) = Symbolics.value(subs_(f, valuation))
eval_(f::Num, val1, val2) = Symbolics.value(subs_(f, eval_(val1, val2)))
# ~~~~

# Vector field of the system
Vfield = [
    x[1] - x[1] * (x[1]^2 + x[2]^2),
    x[2] - x[2] * (x[1]^2 + x[2]^2),
]
println("Vector field: ", Vfield)

# Basis functions for finding the invariants.
# Here all monomials up to degree 2.
basis_funcs = [
    Num(1),
    x[1], x[2],
    x[1]^2, x[1]*x[2], x[2]^2,
]
println("Basis functions: ", basis_funcs)

# Define "Lagrange" multiplier for positivity checking
λ = 1.0
println("λ: ", λ)

# Number of cone refinements = `maxorder + 1` (first corresponds to `0`)
maxorder = 5
println("maxorder: ", maxorder)
# Compute successive application of "∇g⋅f + λg" operator
# for each basis functions up to `maxorder` time.
# Store them in a dictionary with as keys
# the number of application of the operator.
basis_derivs_funcs = Dict{Int,Vector{Num}}()
basis_derivs_funcs[0] = basis_funcs
op_(g) = dot(Symbolics.gradient(g, x), Vfield) + λ * g
for i = 1:maxorder
    old_funcs = basis_derivs_funcs[i - 1]
    @assert length(old_funcs) == length(basis_funcs)
    new_funcs = Vector{Num}(undef, length(basis_funcs))
    for (k, g) in enumerate(old_funcs)
        dg = Symbolics.simplify(op_(g), expand=true)
        new_funcs[k] = dg
    end
    basis_derivs_funcs[i] = new_funcs
end

# Points at which to evaluate the functions stored in `basis_derivs_funcs`
# Here `neval` points on the circle of radius `rad`
neval = 20
rad = 0.5
points_to_eval = [[rad*cos(α), rad*sin(α)]
                  for α in range(0, 2π, neval + 1)[1:neval]]
all_cons = Vector{Float64}[]
for point in points_to_eval
    @assert length(x) == length(point)
    valuation = eval_(point[1], point[2])
    for basis_deriv_funcs in values(basis_derivs_funcs)
        single_con = eval_.(basis_deriv_funcs, (valuation,))
        push!(all_cons, Symbolics.value.(single_con))
    end
end

# Compute the vertices of the polyhedron defined by `all_cons`
verts = Vector{Float64}[]
@static if VERT_ENUM_METH == "CDDLib"
    hr = hrep([HalfSpace(-con, 0) for con in all_cons])
    display(hr)
    feasible_cone = polyhedron(hr, CDDLib.Library())
    @assert isempty(lines(feasible_cone))
    @assert length(points(feasible_cone)) == 1
    verts = collect(ray.a for ray in rays(feasible_cone))
end
@static if VERT_ENUM_METH == "QHull"
    A = zeros(length(all_cons) + 1, length(basis_funcs) + 1)
    for (i, con) in enumerate(all_cons)
        A[i, 1:length(basis_funcs)] = -con
    end
    A[length(all_cons) + 1, 1] = 1.0
    A[length(all_cons) + 1, length(basis_funcs) + 1] = -2.0
    x_int = vcat([1.0], zeros(length(basis_funcs) - 1))
    hs = spatial.HalfspaceIntersection(A, x_int)
    rawverts = collect.(eachrow(hs.intersections))
    ch = spatial.ConvexHull(rawverts)
    verts = [ch.points[i + 1, :] for i in ch.vertices]
    filter!(x -> norm(x) > 1e-6, verts)
    normalize!.(verts)
end
@assert !isempty(verts)
println("# generators after Refinement: ", length(verts))

# Narrowing phase: for each vertex `g` in `current_cone`,
# check if "∇g⋅f + λg" is positive for some λ on the positive set
# defined by `current_cone`.
# If not, remove `g` from `current_cone` and start again.
# Stop when all vertex in `current_cone` are not removed.

# Define variables for SoS.
# Use letter 'd' to indicate different variables than Symbolics
@polyvar xd1 xd2
valuation = eval_(xd1, xd2)

current_cone = BitSet(1:length(verts))
iter = 0
print("Narrowing loop:")

while iter < 100*0
    global iter += 1
    print("\nIter ", iter, ": ")
    i_to_remove = 0
    for i in current_cone
        g_main = dot(verts[i], eval_.(basis_funcs, (valuation,)))
        dg_main = dot(verts[i], eval_.(basis_derivs_funcs[1], (valuation,)))
        S = FullSpace()
        for j in current_cone
            g_con = dot(verts[j], eval_.(basis_funcs, (valuation,)))
            S = S ∩ (SemialgebraicSets.PolynomialInequality(g_con))
        end
        model = SOSModel(Mosek.Optimizer)
        set_silent(model)
        λ = @variable(model)
        con_ref = @constraint(model, dg_main + λ*g_main ≥ 0, domain=S)
        optimize!(model)
        if primal_status(model) != FEASIBLE_POINT
            print("removed: ", i)
            i_to_remove = i
            break
        else
            print(".")
        end
    end
    if i_to_remove > 0
        delete!(current_cone, i_to_remove)
    else
        print("Done")
        break
    end
end
verts = [verts[i] for i in current_cone]
println("\n# generators after Narrowing: ", length(verts))

# Plot the positive set and vector field
vals1_ = range(-2, 2, length=10)
vals2_ = range(-2, 2, length=10)
vals = collect(Iterators.product(vals1_, vals2_))[:]
vals1 = getindex.(vals, 1)
vals2 = getindex.(vals, 2)
flows = [0.04*eval_.(Vfield, val...) for val in vals]
flows1 = getindex.(flows, 1)
flows2 = getindex.(flows, 2)
plt = plot(xlabel="x1", ylabel="x2")
quiver!(vals1, vals2, quiver=(flows1, flows2))

vals1_ = range(-2, 2, length=20)
vals2_ = range(-2, 2, length=20)
Fplot(val1, val2) = (
    basis_vals = eval_.(basis_funcs, val1, val2);
    minimum(coeffs -> dot(coeffs, basis_vals), verts)
)
z = @. Fplot(vals1_', vals2_)
contour!(vals1_, vals2_, z, levels=[0])
display(plt)

end # module