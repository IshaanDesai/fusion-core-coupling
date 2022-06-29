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
    def __init__(self, config, mesh, rho_layers):
        self.rho_min = mesh.get_rhomax() - rho_layers * mesh.get_drho()
        self.rho_max = mesh.get_rhomax()

        self.nrho = mesh.get_nrho()
        self.ntheta = mesh.get_ntheta()
        self.drho = mesh.get_drho()
        self.dtheta = mesh.get_dtheta()

        self.g_rr = mesh.get_g_rho_rho()
        self.g_rt = mesh.get_g_rho_theta()
        self.g_tt = mesh.get_g_theta_theta()

        self.rho = mesh.get_rho_vals()
        self.theta = mesh.get_theta_vals()
        
    def set_bnd_vals_so(self, field, ansol, t, flux=None):
        cdef double [:, ::1] u = field
        # Set the boundary values at the outer edge of the Core domain
        # Pre-compute indices for speed-up
        j = self.nrho-1
        j_m = j-1
        j_mm = j-2

        # Handle periodicity in theta direction due to symmetric stencil
        ip = [self.ntheta - 2, self.ntheta - 1, 0, 1]
        flux_val = 0
        for i in range(1, 3):
            ii = ip[i]
            i_p = ip[i+1]
            i_m = ip[i-1]
            # Dirichlet condition at inner boundary
            u[ii, 0] = ansol.ansol(self.rho[0], self.theta[ii], t)

            if flux is not None:
                flux_val = flux[ii]
            else:
                flux_val = ansol.ansol_gradient(self.rho[j], self.theta[ii], t)

            # Neumann condition at outer boundary (2nd order)
            u[ii, j] = 4 * u[ii, j_m] / 3 - u[ii, j_mm] / 3 - (self.drho * self.g_rt[ii, j]) / (
                3 * self.dtheta * self.g_rr[ii, j]) * (2 * u[i_m, j_m] - 2 * u[i_p, j_m] + u[i_p, j_mm] - u[i_m, j_mm]) + (
                2 * self.drho) / (3 * math.sqrt(self.g_rr[ii, j])) * flux_val

        for i in range(1, self.ntheta - 1):
            i_p = i+1
            i_m = i-1
            i_mm = i-2
            # Dirichlet condition at inner boundary
            u[i, 0] = ansol.ansol(self.rho[0], self.theta[i], t)

            if flux is not None:
                flux_val = flux[i]
            else:
                flux_val = ansol.ansol_gradient(self.rho[j], self.theta[i], t)

            # Neumann condition at outer boundary (2nd order)
            u[i, j] = 4 * u[i, j_m] / 3 - u[i, j_mm] / 3 - (self.drho * self.g_rt[i, j]) / (
                3 * self.dtheta * self.g_rr[i, j]) * (2 * u[i_m, j_m] - 2 * u[i_p, j_m] + u[i_p, j_mm] - u[i_m, j_mm]) + (
                2 * self.drho) / (3 * math.sqrt(self.g_rr[i, j])) * flux_val
        
        return u

    def get_bnd_vals(self, field):
        bnd_data = []
        for j in range(self.nrho):
            for i in range(self.ntheta):
                if self.rho[j] > self.rho_min and self.rho[j] < self.rho_max:
                    bnd_data.append(field[i, j])

        return np.array(bnd_data)

    def get_analytical_bnd_vals(self, ansol, field, t):
        bnd_data = []
        for j in range(self.nrho):
            for i in range(self.ntheta):
                if self.rho[j] > self.rho_min and self.rho[j] < self.rho_max:
                    bnd_data.append(ansol.ansol(self.rho[j], self.theta[i], t))

        return np.array(bnd_data)

    def compare_bnd_flux_vals(self, logger, field, ansol, t, fluxes):
        del2 = 0
        delinf = 0
        # Compare the flux values received at the outer edge of the Core domain with analytical flux values
        j = self.nrho - 1

        for i in range(self.ntheta):
            ansol_flux = ansol.ansol_gradient(self.rho[j], self.theta[i], t)
            # print("Read flux = {}, ansol_flux = {}".format(fluxes[i], ansol_flux))
            del2 += math.pow(fluxes[i] - ansol_flux, 2)
            delinf = max(delinf, abs(fluxes[i] - ansol_flux))

        del2 = math.sqrt(del2 / self.ntheta)

        logger.info("Relative l2 error between mapped fluxes and analytical fluxes = {}".format(del2))
        logger.info("l_inf error between mapped fluxes and analytical fluxes = {}".format(delinf))
