import z3
from sympy import symbols, diff, lambdify
from sympy.utilities.lambdify import lambdastr

def lie_derivative(g, flow, x):
    assert len(flow) == len(x)
    y = 0
    for f, xi in zip(flow, x):
        y += f * diff(g, xi)
    return y

def z3_val_from_sympy(f, x, xz):
    fl = lambdify(x, f)
    return fl(*xz)

def verify_negativity(x, f, gs_neg, gs_zero):
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    xz = [z3.Real("x" + str(k), ctx=ctx) for k in range(len(x))]
    cons = [z3_val_from_sympy(f, x, xz) >= 0]
    for g in gs_neg:
        cons.append(z3_val_from_sympy(g, x, xz) <= 0)
    for g in gs_zero:
        cons.append(z3_val_from_sympy(g, x, xz) == 0)
    solver.add(z3.And(cons, ctx))
    res = solver.check()
    if res == z3.sat:
        model = solver.model()
        print([model[xi] for xi in xz])
    else:
        print("unsat")

def verify_invariance(flow, gs, x):
    for g in gs:
        dg = lie_derivative(g, flow, x)
        gs_zero = [g]
        gs_neg = []
        for h in gs:
            if h == g:
                continue
            else:
                gs_neg.append(h)
        verify_negativity(x, dg, gs_neg, gs_zero)


if __name__ == "__main__":
    x0, x1 = symbols('x0 x1')
    x = [x0, x1]

    f = x0**2 - x1**2
    gs_neg = []
    gs_zero = []
    gs_neg.append(0.5**2 - x1**2)
    gs_zero.append(0.5 * x0 - x0**2)

    verify_negativity(x, f, gs_neg, gs_zero)

    gs_zero.append(0.5**2 - x0**2)
    
    verify_negativity(x, f, gs_neg, gs_zero)

    gs_neg.append(0.9 - x1)

    verify_negativity(x, f, gs_neg, gs_zero)

    print("-" * 80)

    flow = [
        2.0*x1 - 0.5*x0,
        -0.5*x1,
    ]
    g = 1 - x0 + x1
    dg = lie_derivative(g, flow, x)
    print(dg)
    
    print("-" * 80)