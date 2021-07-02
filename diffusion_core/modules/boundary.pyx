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
        # Pre-compute indices for speed-up
        j = self.nrho-1
        j_m = j-1
        j_mm = j-2

        # Handle periodicity in theta direction due to symmetric stencil
        ip = [self.ntheta - 2, self.ntheta - 1, 0, 1]
        for i in range(1, 3):
            ii = ip[i]
            i_p = ip[i+1]
            i_m = ip[i-1]
            # Dirichlet condition at inner boundary
            field[ii, 0] = ansol.ansol(self.rho[0], self.theta[ii], t)
            # Neumann condition at outer boundary (2nd order)
            field[ii, j] = 4 * field[ii, j_m] / 3 - field[ii, j_mm] / 3 - (self.drho * self.g_rt[ii, j]) / (
                3 * self.dtheta * self.g_rr[ii, j]) * (2 * field[i_m, j_m] - 2 * field[i_p, j_m] + field[i_p, j_mm] - field[i_m, j_mm]) + (
                2 * self.drho) / (3 * math.sqrt(self.g_rr[ii, j])) * (flux[ii])

        for i in range(1, self.ntheta - 1):
            i_p = i+1
            i_m = i-1
            i_mm = i-2
            # Dirichlet condition at inner boundary
            field[i, 0] = ansol.ansol(self.rho[0], self.theta[i], t)
            # Neumann condition at outer boundary (2nd order)
            field[i, j] = 4 * field[i, j_m] / 3 - field[i, j_mm] / 3 - (self.drho * self.g_rt[i, j]) / (
                3 * self.dtheta * self.g_rr[i, j]) * (2 * field[i_m, j_m] - 2 * field[i_p, j_m] + field[i_p, j_mm] - field[i_m, j_mm]) + (
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
        j_m = j-1
        j_mm = j-2

        # Handle periodicity in theta direction due to symmetric stencil
        ip = [self.ntheta - 2, self.ntheta - 1, 0, 1]
        for i in range(1, 3):
            ii = ip[i]
            i_p = ip[i+1]
            i_m = ip[i-1]
            # Dirichlet condition at inner boundary
            field[ii, 0] = ansol.ansol(self.rho[0], self.theta[ii], t)
            # Neumann condition at outer boundary (2nd order)
            flux = ansol.ansol_gradient(self.rho[j], self.theta[ii], t)
            field[ii, j] = 4 * field[ii, j_m] / 3 - field[ii, j_mm] / 3 - (self.drho * self.g_rt[ii, j]) / (
                3 * self.dtheta * self.g_rr[ii, j]) * (2 * field[i_m, j_m] - 2 * field[i_p, j_m] + field[i_p, j_mm] - field[i_m, j_mm]) + (
                2 * self.drho) / (3 * math.sqrt(self.g_rr[ii, j])) * flux

        for i in range(1, self.ntheta - 1):
            i_p = i+1
            i_m = i-1
            i_mm = i-2
            # Dirichlet condition at inner boundary
            field[i, 0] = ansol.ansol(self.rho[0], self.theta[i], t)
            # Neumann condition at outer boundary (2nd order)
            flux = ansol.ansol_gradient(self.rho[j], self.theta[i], t)
            field[i, j] = 4 * field[i, j_m] / 3 - field[i, j_mm] / 3 - (self.drho * self.g_rt[i, j]) / (
                3 * self.dtheta * self.g_rr[i, j]) * (2 * field[i_m, j_m] - 2 * field[i_p, j_m] + field[i_p, j_mm] - field[i_m, j_mm]) + (
                2 * self.drho) / (3 * math.sqrt(self.g_rr[i, j])) * flux
