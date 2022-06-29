"""
Analytical solution module for the diffusion equation using Bessel functions
"""

import numpy as np
cimport numpy as np
cimport cython
import scipy.special as sp
import math


cdef class Ansol:
    def __init__(self, config, mesh):
        self.m = config.get_pol_mode_number()  # Poloidal mode number
        self.ums = config.get_ums()  # 2nd (radial mode num) zero of Bessel function of first kind of order m,
        # see https://mathworld.wolfram.com/BesselFunctionZeros.html

        self.rho, self.theta = mesh.get_rho_vals(), mesh.get_theta_vals()
        self.nrho, self.ntheta = mesh.get_nrho(), mesh.get_ntheta()
        self.drho, self.dtheta = mesh.get_drho(), mesh.get_dtheta()

    def ansol(self, r, theta, t):
        return math.sin(self.m * theta) * sp.jv(self.m, self.ums * r) * math.exp(-math.pow(self.ums, 2) * t)

    def ansol_gradient(self, r, theta, t):
        return math.sin(self.m * theta) * math.exp(-math.pow(self.ums, 2) * t) * self.ums * sp.jvp(self.m, self.ums * r)

    def compare_ansoln(self, u, t, logger):
        del2 = 0
        delinf = 0
        soln = 0
        for i in range(self.ntheta):
            for j in range(self.nrho):
                del2 += math.pow(u[i, j] - self.ansol(self.rho[j], self.theta[i], t), 2)
                soln += math.pow(self.ansol(self.rho[j], self.theta[i], t), 2)
                delinf = max(delinf, abs(u[i, j] - self.ansol(self.rho[j], self.theta[i], t)))

        del2 = math.sqrt(del2 / soln)

        logger.info("The l2 error between numerical and analytical solution is {}".format(del2))
        logger.info("The l-inf (max) error between numerical and analytical solution is {}".format(delinf))
