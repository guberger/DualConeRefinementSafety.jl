module Example

using LaTeXStrings
using DynamicPolynomials
using SumOfSquares
using Plots

include("../../src/InvariancePolynomial.jl")
const MP = InvariancePolynomial.Projection

var, = @polyvar x[1:2]
funcs = [1, x[1]^2, x[1]*x[2], x[2]^2]
dom = @set (
    -0.5 * x[1]^2 - 0.5 * x[1]*x[2] + 0.5 * x[2]^2 ≤ 1 &&
    0.5 * x[1]^2 - 0 * x[1]*x[2] + 0.25 * x[2]^2 ≤ 1
)

plt = plot(xlabel=L"x_1", ylabel=L"x_2",
           aspect_ratio=:equal,
           xlims=(-2.5, 2.5), ylims=(-2.5, 2.5),
           dpi=400)

x1s_ = range(-2.5, 2.5, length=500)
x2s_ = range(-2.5, 2.5, length=500)
Fplot_init(x1, x2) = minimum(g(var=>[x1, x2]) for g in inequalities(dom))
z = @. Fplot_init(x1s_', x2s_)
contourf!(x1s_, x2s_, z, levels=[0, 100],
          lw=0, c=:green, fa=0.2, colorbar=:none)
contour!(x1s_, x2s_, z, levels=[0], lw=4, c=:green, colorbar=:none)
for g in inequalities(dom)
    Fplot_init(x1, x2) = g(var=>[x1, x2])
    z = @. Fplot_init(x1s_', x2s_)
    contour!(x1s_, x2s_, z, levels=[0],
             lw=1, c=:black, colorbar=:none)
end

savefig(plt, "examples/figures/bsa_set.png")

end # module