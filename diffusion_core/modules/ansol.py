"""
Analytical solution module for the diffusion equation using Bessel functions
"""

import numpy as np
from scipy import special
import math

class Ansol:
    def __init__(self, config, mesh):
        self._m = config.get_pol_mode_number() # Poloidal mode number
        self._ums = config.get_ums() # 2nd (radial mode num) zero of Bessel function of first kind of order m, see https://mathworld.wolfram.com/BesselFunctionZeros.html 

        self._diffc = config.get_diffusion_coeff()
        self._rho, self._theta = mesh.get_rho_vals(), mesh.get_theta_vals()
        self._nrho, self._ntheta = mesh.get_nrho(), mesh.get_ntheta()
        self._drho, self._dtheta = mesh.get_drho(), mesh.get_dtheta()

    def ansol(self, r, theta, t):
        return math.sin(self._m*theta)*special.jv(self._m,self._ums*r)*math.exp(-self._diffc*math.pow(self._ums,2)*t)

    def ansol_gradient(self, r, theta, t):
        return math.sin(self._m*theta)*math.exp(-self._diffc*math.pow(self._ums,2)*t)*self._ums*(1/2) * \
            (special.jv(self._m-1,self._ums*r) - special.jv(self._m+1,self._ums*r))

    def compare_ansoln(self, u, t, logger_ansoln):
        del2 = 0
        delinf = 0
        for i in range(self._ntheta):
            for j in range(self._nrho):
                del2 += self._drho*self._dtheta*self._rho[j]*math.pow(u[i, j] - self.ansol(self._rho[j], self._theta[i], t), 2)
                delinf = max(delinf, abs(u[i, j] - self.ansol(self._rho[j], self._theta[i], t)))
        
        del2 = math.sqrt(del2)

        logger_ansoln.info("The l2 error between numerical and analytical solution is {}".format(del2))
        logger_ansoln.info("The l-inf (max) error between numerical and analytical solution is {}".format(delinf))
