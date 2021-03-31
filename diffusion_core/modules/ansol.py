"""
Analytical solution module for the diffusion equation using Bessel functions
"""

import numpy as np
from scipy import special
import math

class Ansol:

    def __init__(self, config):
        self._m = 2 # Poloidal mode number
        self._s = 2 # Radial mode number
        self._ums = 8.4172 # s'th zero of Bessel function of first kind of order m, see https://mathworld.wolfram.com/BesselFunctionZeros.html 

        self._diffc = config.get_diffusion_coeff()

    def ansol(self, r, theta, t):
        return math.sin(self._m*theta)*special.jv(self._m,self._ums*r)*math.exp(-self._diffc*math.pow(self._ums,2)*t)

    def compare_ansoln(self, mesh, u, t, logger_ansoln):
        del2 = 0
        delinf = 0
        for l in range(mesh.get_n_points_mesh()):
            i, j = mesh.get_i_j_from_index(l)
            r = mesh.get_r(l)
            theta = mesh.get_theta(l)
            dr, dtheta = mesh.get_r_spacing(), mesh.get_theta_spacing()

            del2 += dr*dtheta*r*math.pow(u[i, j] - self.ansol(r, theta, t), 2)
            delinf = max(delinf, abs(u[i, j] - self.ansol(r, theta, t)))
        
        del2 = math.sqrt(del2)

        logger_ansoln.info("The l2 error between numerical and analytical solution is {}".format(del2))
        logger_ansoln.info("The l-inf (max) error between numerical and analytical solution is {}".format(delinf))
