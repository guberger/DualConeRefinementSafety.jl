import sys
import os
sys.path.append(os.path.abspath(__file__) + '/../../../')
from src.verify import verify_invariance

from sympy import symbols, lambdify
import z3

x0, x1, x2 = symbols('x0 x1 x2')
x = [x0, x1]

# Run your favorite exa*.jl file
# Copy-paste the output below and replace hat by double asterisk
# Create variables x0, ..., xn as needed

flow = [
x1,
0.5*x1 - x0 - 0.5*x0**2*x1,
]

gs = [
-0.9930014364500316 + 0.03854809753170376*x1**2 - 0.06747618398702622*x0*x1 - 0.08893343566628577*x0**2,
-0.9992392424888992 + 0.011227285361442449*x1**2 + 0.0022624216704171473*x0*x1 + 0.03727956252169085*x0**2,
-0.9984529561179857 + 0.03358278465026513*x1**2 - 0.036674606788979706*x0*x1 - 0.024876981554505257*x0**2,
]

verify_invariance(flow, gs, x)