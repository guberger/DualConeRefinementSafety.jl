import sys
import os
sys.path.append(os.path.abspath(__file__) + '/../../../')
from src.verify import verify_invariance

from sympy import symbols, lambdify
import z3

x0, x1 = symbols('x0 x1')
x = [x0, x1]

# Run your favorite exa*.jl file
# Copy-paste the output below and replace hat by double asterisk
# Create variables x0, ..., xn as needed

flow = [
x1,
0.5*x1 - x0 - 0.5*x0**2*x1,
]

gs = [
-0.993031954986876 + 0.03778153849889998*x1**2 - 0.0666256032861187*x0*x1 - 0.08956070963519557*x0**2,
-0.9991994975155781 + 0.007562020589211854*x1**2 + 0.004248519887888844*x0*x1 + 0.039052913950008275*x0**2,
-0.9993785554131969 + 0.01537369337658755*x1**2 - 0.0017643808674354974*x0*x1 + 0.03167079873241948*x0**2,
]

verify_invariance(flow, gs, x)