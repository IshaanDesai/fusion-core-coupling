"""
This module consists of functionality relevant for a Method of Manufactured Soltuions analysis of the diffusion code
"""
import math


def source_term(rmin, rmax, r, theta, t):
    """
    Calculation of source term based on assumed solution: f = sin(2*pi*(r - rmin)/(rmax - rmin))*cost(omega*t)*cos(theta)
    This solution is plugged into the equation S_mms = dt(f) - D*dperp^2(f)
    S_mms is the pseudo source term generated
    This source term is added to the current finite difference stencil evaluation
    :return:
    """
    var1 = 2*math.pi*(r - rmin)/(rmax - rmin)
    var2 = 2*math.pi/(rmax - rmin)
    source = -math.sin(var1)*math.sin(t)*math.cos(theta)
    source += -diffc*math.cos(t)*math.cos(theta)*(-(var2**2)*math.sin(t) + var2*math.cos(var1) - math.sin(var1))
