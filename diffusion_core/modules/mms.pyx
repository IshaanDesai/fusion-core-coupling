"""
This module consists of functionality relevant for a Method of Manufactured Soltuions analysis of the diffusion code
"""
import numpy as np
cimport numpy as np
cimport cython
import math


cdef class MMS:
    """
    Enable "Method of Manufactured Solutions" to check code
    """
    def __init__(self, config, mesh):
        """
        :param config: Configuration module object
        """

        self.rmin = config.get_rmin()
        self.rmax = config.get_rmax()
        self.diffc = config.get_diffusion_coeff()

        self.nr, self.ntheta = mesh.get_n_points_axiswise()

        self.dr = mesh.get_r_spacing()
        self.dtheta = 2 * math.pi / config.get_theta_points()

    def init_mms(self, r, theta):
        """
        Initial state based on the mms solution f = sin(2*pi*(r - rmin)/(rmax - rmin))*cost(omega*t)*cos(theta)
        Achieved by inserting t = 0 in the solution.
        :param mesh:
        :param r:
        :param theta:
        :return:
        """
        return math.sin(2 * math.pi * (r - self.rmin) / (self.rmax - self.rmin)) * math.cos(theta)

    def source_term(self, r, theta, t):
        """
        Calculation of source term based on assumed solution: f = sin(2*pi*(r - rmin)/(rmax - rmin))*cos(t)*cos(theta)
        This solution is plugged into the equation S_mms = dt(f) - D*dperp^2(f)
        S_mms is the pseudo source term generated
        This source term is added to the current finite difference stencil evaluation
        :return:
        """
        cdef double var1 = 2 * math.pi * (r - self.rmin) / (self.rmax - self.rmin)
        cdef double var2 = 2 * math.pi / (self.rmax - self.rmin)
        cdef double source = -math.sin(var1) * math.sin(t) * math.cos(theta)
        source += -self.diffc * math.cos(t) * math.cos(theta) * (
                    -(var2 ** 2) * math.sin(t) + var2 * math.cos(var1) - math.sin(var1))

        return source

    def analytical_soln(self, r, theta, t):
        """
        Solution f = sin(2*pi*(r - rmin)/(rmax - rmin))*cos(t)*cos(theta)
        :param r:
        :param theta:
        :param t:
        :return:
        """
        var1 = 2*math.pi*(r - self.rmin)/(self.rmax - self.rmin)
        return math.sin(var1)*math.cos(t)*math.cos(theta)

    def error_computation(self, mesh, field, t):
        """
        Calculate analytical solution using Ansatz selected before and calculate L2 error with respect to solution
        obtained by the discretized stencil
        :return:
        """
        error_abs, mms_sum = 0, 0
        for i in range(1, self._nr - 1):
            for j in range(1, self.ntheta + 1):
                mesh_ind = mesh.get_index_from_i_j(i, j - 1)
                r = mesh.get_r(mesh_ind)
                theta = mesh.get_theta(mesh_ind)
                error_abs += abs(field[i, j] - self.analytical_soln(r, theta, t))**2 * r * self._dr * self._dtheta
                mms_sum += self.analytical_soln(r, theta, t)**2 * r * self._dr * self._dtheta

        return error_abs/mms_sum




