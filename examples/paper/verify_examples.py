import sys
import os
sys.path.append(os.path.abspath(__file__) + '/../../../')
from src.verify import verify_invariance

from sympy import symbols, lambdify
import z3

x0, x1, x2 = symbols('x0 x1 x2')
x = [x0, x1, x2]

# Run your favorite exa*.jl file
# Copy-paste the output below and replace hat by double asterisk
# Create variables x0, ..., xn as needed

flow = [
1 - x0*x2**2 - x0**3 - x0*x2**4 - x0**3*x2**2,
-x1 - x1*x2**2 - x0**2*x1 - x0**2*x1*x2**2,
-4*x2 - x2**3 + 3*x0**2*x2 + 3*x0**2*x2**3,
]

gs = [
-0.871872297547406 - 0.21177461319034813*x2**2 + 0.10501545362584444*x1**2 + 0.42890787411429493*x0**2,
# -0.8535189933667975 - 0.17888097824420462*x2**2 + 0.007807118789072992*x1**2 + 0.4893321698812969*x0**2,
# -0.8650501279681401 - 0.200096067640876*x2**2 + 0.0877118073301994*x1**2 + 0.45161541013547407*x0**2,
# -0.6906501877035409 + 0.6121202891787498*x2**2 - 0.0030550649675006266*x1**2 + 0.38509964473994507*x0**2,
-0.3329552584068616 + 0.9316113015966564*x2**2 + 0.02899739413846004*x1**2 + 0.1428297229913382*x0**2,
# -0.33337674127559513 + 0.9332427750444002*x2**2 - 0.0032308722117315894*x1**2 + 0.13381865590657924*x0**2,
# -0.33307532808420715 + 0.9321483462528075*x2**2 - 0.0022873854889847497*x1**2 + 0.14196849744707027*x0**2,
]

verify_invariance(flow, gs, x)