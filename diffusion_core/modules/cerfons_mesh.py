"""

"""

import numpy as np
from numpy import pi, cos, sin, cosh, sinh, log, sqrt
import math

class meshCERFONS:
    """
    Mesh of type CERFONS (https://aip.scitation.org/doi/abs/10.1063/1.3328818?journalCode=php)
    Constants and code for PSI provided by Thomas Body (Max Planck Institute of Plasma Physics)
    """

    def __init__(self, nrho, ntheta, rhomin, rhomax):
        """
        Constructor for object of type meshCERFONS
        """

        # Pre-defined constants for CERFONS
        self._x0 = 1.0
        self._y0 = 0.0
        self._cerf_a = 0.0
        self._cerf_c1 = 0.0735011444550040364542354505071
        self._cerf_c2 = -0.0866241743631724894540195833599
        self._cerf_c3 = -0.146393154340110145566533995171
        self._cerf_c4 = -0.0763123710053627236722106917166
        self._cerf_c5 = 0.0903179011379421292953276540381
        self._cerf_c6 = -0.0915754123901870980044268057017
        self._cerf_c7 = -0.00389228297983755811356246513352
        self._cerf_c8 = 0.0427189122507639110238025727235
        self._cerf_c9 = 0.227554564600278024358552580518
        self._cerf_c10 = -0.130472413601776209642913874538
        self._cerf_c11 = -0.0300697410847693691591393359449
        self._cerf_c12 = 0.00421267189210391228482454852824
        self._cerf_r0 = 1.073539190422562
        self._cerf_z0 = 0.029233201183845
        self._cerf_psio = -0.058149455020455
        self._cerf_psix = 0.0

    def _psi_cerfons(self, x, y):
        rtemp = x * self._cerf_r0
        ztemp = self._cerf_r0 * (y + self._cerf_z0 / self._cerf_r0)

        psi = (rtemp ** 4 / 8.0 + cerf_a * (rtemp ** 2 * log(rtemp) / 2.0 - rtemp ** 4 / 8.0)
               + self._cerf_c1 * 1.0
               + self._cerf_c2 * (rtemp ** 2)
               + self._cerf_c3 * (ztemp ** 2 - rtemp ** 2 * log(rtemp))
               + self._cerf_c4 * (rtemp ** 4 - 4.0 * rtemp ** 2 * ztemp ** 2)
               + self._cerf_c5 * (2. * ztemp ** 4 - 9. * ztemp ** 2 * rtemp ** 2 + (
                        3. * rtemp ** 4 - 12. * rtemp ** 2 * ztemp ** 2) * log(rtemp))
               + self._cerf_c6 * (rtemp ** 6 - 12.0 * rtemp ** 4 * ztemp ** 2 + 8.0 * rtemp ** 2 * ztemp ** 4)
               + self._cerf_c7 * (8.0 * ztemp ** 6 - 140.0 * ztemp ** 4 * rtemp ** 2 + 75.0 * ztemp ** 2 * rtemp ** 4
                                  + (-15.0 * rtemp ** 6 + 180.0 * rtemp ** 4 * ztemp ** 2 - 120.0 * rtemp ** 2 *
                                     ztemp ** 4) * log(rtemp))
               + self._cerf_c8 * ztemp
               + self._cerf_c9 * (rtemp ** 2 * ztemp)
               + self._cerf_c10 * (ztemp ** 3 - 3.0 * rtemp ** 2 * ztemp * log(rtemp))
               + self._cerf_c11 * (3.0 * rtemp ** 4 * ztemp - 4.0 * rtemp ** 2 * ztemp ** 3)
               + self._cerf_c12 * (8.0 * ztemp ** 5 - 45.0 * rtemp ** 4 * ztemp
                                   + (-80.0 * rtemp ** 2 * ztemp ** 3 + 60.0 * rtemp ** 4 * ztemp) * log(rtemp))
               )

        return psi

    def _get_rho(self, x, y):
        """
        Calculate rho coordinate value for a given Cartesian coordinate
        """
        return sqrt((self._psi_cerfons(x, y) - self._cerf_psix) / (self._cerf_psio - self._cerf_psix))

    def _get_theta(self, x, y):
        """
        Calculate theta coordinate value for a given Cartesian coordinate
        """
        if x > 0.0:
            theta = math.atan(y/x)
        elif x < 0.0:
            theta = math.atan(y/x) + math.pi
        else:
            if y > 0.0:
                theta = math.pi/2.0
            elif y < 0.0:
                theta = math.pi*3.0/2.0
            else:
                theta = 0.0

        return theta

    def get_jacobian(self, x, y):
        """
        Get determinant of Jacobian matrix of transformation from Toroidal-like coordinates to Cartesian coordinates
        """
        g_rr, g_tt, g_pp, g_rt = self.get_metric_coeffs(x, y)

        return sqrt(1.0 / (g_rr*g_tt*g_pp - g_pp*g_rt*g_rt))

    def get_metric_coeffs(self, x, y):
        """
        Get the metric coefficient values for a given Cartesian coordinate
        The metric coefficients are ordered as follows: g_rhorho, g_thetatheta, g_phiphi, g_rhotheta
        """
        eps_findiff = 1.0E-6
        xa = x - self._x0
        ya = y - self._y0

        # g_thetatheta
        g_tt = 1.0 / (xa**2 + ya**2)

        drhodx = (self._get_rho(x+eps_findiff, y) - self._get_rho(x-eps_findiff, y)) / (2.0*eps_findiff)
        drhody = (self._get_rho(x, y+eps_findiff) - self._get_rho(x, y-eps_findiff)) / (2.0*eps_findiff)

        # g_rhorho
        g_rr = drhodx**2 + drhody**2

        # g_rhotheta
        g_rt = (-ya*drhodx + xa*drhody) / (xa**2 + ya**2)

        # g_phiphi
        g_pp = 0.0

        return g_rr, g_tt, g_pp, g_rt


