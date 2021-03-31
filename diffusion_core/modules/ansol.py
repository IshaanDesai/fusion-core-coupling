"""
Analytical solution module for the diffusion equation using Bessel functions
"""

import numpy as np
from scipy import special
import math

class Ansol:

    def __init__(self):
        self._m = 2 # Poloidal mode number
        self._s = 2 # Radial mode number
        self._ums = 8.4172 # s'th zero of Bessel function of first kind of order m, see https://mathworld.wolfram.com/BesselFunctionZeros.html 

    def ansol(self, r, theta, t):
        return math.sin(self._m*theta)*special.jv(1,self._ums*r)*math.exp(math.pow(-self._ums,2)*t)

    def compare_ansoln(self, mesh, u, t):
        del2 = 0
        for l in range(mesh.get_n_points_mesh()):
            i, j = mesh.get_i_j_from_index(l)
            r = mesh.get_r(l)
            theta = mesh.get_theta(l)
            del2 += math.pow(u[i, j] - self.ansol(r, theta, t), 2)
        
        del2 = math.sqrt(del2)

        print("The l2 error between numerical and analytical solution is {}".format(del2))
