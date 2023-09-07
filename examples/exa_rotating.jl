module ExampleRotating

using LinearAlgebra
using Symbolics
using Polyhedra
using CDDLib
using MosekTools
using SumOfSquares
using Plots

solver() = optimizer_with_attributes(Mosek.Optimizer, "QUIET"=>true)

include("../src/DualConeRefinementSafety.jl")
const DCR = DualConeRefinementSafety

# Create the variables for symbolic manipulation
Symbolics.@variables x1, x2
vars = [x1, x2]
f = [
    +x2 + x1 - x1 * (x1^2 + x2^2),
    -x1 + x2 - x2 * (x1^2 + x2^2),
]
tmp = DCR.Template(vars, [1, x1^2, x1*x2, x2^2])
λ = 1
maxorder = 5

np = 10
rad = 0.5
points = [[rad*cos(α), rad*sin(α)] for α in range(0, 2π, np + 1)[1:np]]
hc = DCR.hcone_from_points(tmp, f, λ, maxorder, points)
display(length(hc.supps))
vc = DCR.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.gens))

function callback_func(iter1, iter2, ::Any, indices, is_good)
    nvert = length(indices)
    if is_good && iter2 < nvert
        if mod(iter2, 10) == 0
            print("Iter $(iter1): $(iter2) / $(nvert) \r")
        end
    elseif iter2 == nvert
        print("Iter $(iter1): $(iter2) / $(nvert) -> valid\n")
    else
        print("Iter $(iter1): $(iter2) / $(nvert) -> bad found\n")
    end
    return nothing
end
vc, success = DCR.narrow_vcone(vc, f, Inf, solver,
                               callback_func=callback_func)
println(success)

# Plot the positive set and vector field
x1s_ = range(-2, 2, length=10)
x2s_ = range(-2, 2, length=10)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
subs_ = Symbolics.substitute
eval_(x) = Dict(zip(vc.tmp.vars, x))
eval_(f::Num, x) = Symbolics.value(subs_(f, eval_(x)))
dxs = [0.04*eval_.(f, (x,)) for x in xs]
dxs1 = getindex.(dxs, 1)
dxs2 = getindex.(dxs, 2)
plt = plot(xlabel="x1", ylabel="x2")
quiver!(x1s, x2s, quiver=(dxs1, dxs2))

x1s_ = range(-2, 2, length=20)
x2s_ = range(-2, 2, length=20)
Fplot(x1, x2) = (fs = eval_.(vc.tmp.funcs, ((x1, x2),));
                 maximum(c -> dot(c, fs), vc.gens))
z = @. Fplot(x1s_', x2s_)
contour!(x1s_, x2s_, z, levels=[0])
display(plt)

end # module