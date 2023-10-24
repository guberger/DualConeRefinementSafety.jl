import sys
import os
sys.path.append(os.path.abspath(__file__) + '/../../')
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
-0.9805776387586669 + 0.07545672968439855*x1**2 - 0.1429278230690141*x0*x1 - 0.1111099172196857*x0**2,
-0.9872780013777702 + 0.07687802454126674*x1**2 - 0.1175968505984644*x0*x1 - 0.07445064182040449*x0**2,
-0.9972696988976962 + 0.021556474042977804*x1**2 + 0.0044804344770096255*x0*x1 + 0.07048682000367082*x0**2,
-0.997275242281236 + 0.023471115088188818*x1**2 + 0.0035150998492760816*x0*x1 + 0.06984870766498849*x0**2,
-0.9972682239445492 + 0.021793904935726156*x1**2 + 0.004373385078427899*x0*x1 + 0.07044138500265797*x0**2,
-0.997338849794285 + 0.0277497542224067*x1**2 + 0.0006667069091546601*x0*x1 + 0.06741457804886081*x0**2,
-0.9974458278837872 + 0.031816970759395345*x1**2 - 0.002912192847343053*x0*x1 + 0.06388286109545839*x0**2,
-0.9975910598275205 + 0.03843605243173237*x1**2 - 0.009662235498939917*x0*x1 + 0.05693319269839226*x0**2,
-0.9955492918996736 + 0.06851671340624609*x1**2 - 0.06447369083077333*x0*x1 - 0.005496414533368448*x0**2,
-0.996167267581362 + 0.06601320736114238*x1**2 - 0.05730009884128467*x0*x1 + 0.003119315014037795*x0**2,
-0.9949816177445326 + 0.07022788001581634*x1**2 - 0.07017226666095631*x0*x1 - 0.0124690902082031*x0**2,
]

verify_invariance(flow, gs, x)