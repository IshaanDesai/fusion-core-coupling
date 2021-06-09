"""
This module implements boundary conditions on the boundary points of a given mesh.
Dirichlet and Neumann boundary conditions can be implemented.
For Neumann boundary conditions, the flux normal is in the inward directions (towards the center of polar grid).

NOTE: Neumann boundary conditions are implemented with the assumption that only Wall Boundary points will have
      Neumann conditions.
"""
import numpy as np
cimport numpy as np
cimport cython
import math


cdef class Boundary:
    def __init__(self, mesh):
        self.nrho = mesh.get_nrho()
        self.ntheta = mesh.get_ntheta()
        self.drho = mesh.get_drho()
        self.dtheta = mesh.get_dtheta()
        self.g_rr = mesh.get_g_rho_rho()
        self.g_rt = mesh.get_g_rho_theta()
        self.g_tt = mesh.get_g_theta_theta()
        self.rho = mesh.get_rho_vals()
        self.theta = mesh.get_theta_vals()

    def set_bnd_vals_so(self, field, ansol, t, flux):
        # Set the boundary values at the outer edge of the Core domain
        j = self.nrho - 1

        # Handle periodicity in theta direction due to symmetric stencil
        ip = [self.ntheta - 2, self.ntheta - 1, 0, 1]
        for i in range(1, 3):
            # Dirichlet condition at inner boundary
            field[ip[i], 0] = ansol.ansol(self.rho[0], self.theta[ip[i]], t)
            # Neumann condition at outer boundary (2nd order)
            field[ip[i], j] = 4 * field[ip[i], j - 1] / 3 - field[ip[i], j - 2] / 3 - (
                    self.drho * self.g_rt[ip[i], j]) / (3 * self.dtheta * self.g_rr[ip[i], j]) * (
                                      2 * field[ip[i - 1], j - 1] - 2 * field[ip[i + 1], j - 1] + field[
                                  ip[i + 1], j - 2] -
                                      field[ip[i - 1], j - 2]) + (2 * self.drho) / (
                                      3 * math.sqrt(self.g_rr[ip[i], j])) * (flux[ip[i]])

        for i in range(1, self.ntheta - 1):
            # Dirichlet condition at inner boundary
            field[i, 0] = ansol.ansol(self.rho[0], self.theta[i], t)
            # Neumann condition at outer boundary (2nd order)
            field[i, j] = 4 * field[i, j - 1] / 3 - field[i, j - 2] / 3 - (self.drho * self.g_rt[i, j]) / (
                    3 * self.dtheta * self.g_rr[i, j]) * (2 * field[i - 1, j - 1] - 2 * field[i + 1, j - 1] +
                                                            field[i + 1, j - 2] - field[i - 1, j - 2]) + (
                                  2 * self.drho) / (3 * math.sqrt(self.g_rr[i, j])) * (flux[i])

    def get_bnd_vals(self, field):
        bnd_data = []
        # Write data from the interior of the domain (2 mesh widths inside the physical boundary)
        #j = self._nrho - 3
        write_polar_range = [self.nrho-4, self.nrho-3, self.nrho-2]
        # Gets Dirichlet values and returns them for coupling
        for j in write_polar_range:
            for i in range(self.ntheta):
                bnd_data.append(field[i, j])

        return np.array(bnd_data)

    def set_bnd_vals_ansol(self, field, ansol, t):
        """
        Apply Dirichlet boundary condition of value of analytical solution at inner boundary
        Apply Neumann boundary condition of flux of gradient of analytical solution at outer boundary
        """
        # Set the boundary values at the outer edge of the Core domain
        j = self.nrho - 1

        # Handle periodicity in theta direction due to symmetric stencil
        ip = [self.ntheta - 2, self.ntheta - 1, 0, 1]
        for i in range(1, 3):
            # Dirichlet condition at inner boundary
            field[ip[i], 0] = ansol.ansol(self.rho[0], self.theta[ip[i]], t)
            # Neumann condition at outer boundary (2nd order)
            flux = ansol.ansol_gradient(self.rho[j], self.theta[ip[i]], t)
            field[ip[i], j] = 4 * field[ip[i], j - 1] / 3 - field[ip[i], j - 2] / 3 - (
                    self.drho * self.g_rt[ip[i], j]) / (3 * self.dtheta * self.g_rr[ip[i], j]) * (
                                      2 * field[ip[i - 1], j - 1] - 2 * field[ip[i + 1], j - 1] +
                                      field[ip[i + 1], j - 2] - field[ip[i - 1], j - 2]) + (
                                      2 * self.drho) / (3 * math.sqrt(self.g_rr[ip[i], j])) * flux

        for i in range(1, self.ntheta - 1):
            # Dirichlet condition at inner boundary
            field[i, 0] = ansol.ansol(self.rho[0], self.theta[i], t)
            # Neumann condition at outer boundary (2nd order)
            flux = ansol.ansol_gradient(self.rho[j], self.theta[i], t)
            field[i, j] = 4 * field[i, j - 1] / 3 - field[i, j - 2] / 3 - (self.drho * self.g_rt[i, j]) / (
                    3 * self.dtheta * self.g_rr[i, j]) * (
                                  2 * field[i - 1, j - 1] - 2 * field[i + 1, j - 1] + field[i + 1, j - 2] -
                                  field[i - 1, j - 2]) + (2 * self.drho) / (3 * math.sqrt(self.g_rr[i, j])) * flux


    def set_bnd_vals_ansol_outer_dir(self, field, ansol, t):
        """
        Set outer boundary to analytical solution Dirichlet values.
        This function is used for uni-directional coupling
        """
        # Set the boundary values at the outer edge of the Core domain
        j = self.nrho - 1

        # Handle periodicity in theta direction due to symmetric stencil
        ip = [self.ntheta - 2, self.ntheta - 1, 0, 1]
        for i in range(1, 3):
            # Dirichlet condition at inner boundary
            field[ip[i], 0] = ansol.ansol(self.rho[0], self.theta[ip[i]], t)
            # Dirichlet condition at outer boundary
            field[ip[i], j] = ansol.ansol(self.rho[j], self.theta[ip[i]], t)

        for i in range(1, self.ntheta - 1):
            # Dirichlet condition at inner boundary
            field[i, 0] = ansol.ansol(self.rho[0], self.theta[i], t)
            # Dirichlet condition at outer boundary
            field[i, j] = ansol.ansol(self.rho[j], self.theta[i], t)
